"""
Ray Train recipe for PC-RF: data-parallel (DDP) training that scales from a
laptop to a Slurm cluster (Ray-on-Slurm) to a managed Ray service with ONE entrypoint. The backend is
chosen by how Ray is initialized, not by editing this file:

  local    (CPU/synthetic):  python train_ray.py --data synthetic --workers 2 --epochs 1
  Slurm    (Ray-on-Slurm):   srun ... then python train_ray.py --data era5 --gpu --workers $N
  Managed  (e.g. Anyscale):  submit this script as the job entrypoint

Real ERA5 training, fault-tolerant checkpointing (resumes after preemption), and
per-epoch throughput so the 1/2/4/8 GPU scaling curve has real samples/sec.

Ingest backends (--ingest, default "torch", fully backward-compatible):
  torch : the original path. ray.train.torch.prepare_data_loader wraps a plain
          torch DataLoader. For ERA5 this forces num_workers=0 because
          ERA5Dataset holds non-fork-safe netCDF handles.
  ray   : a Ray Data pipeline. Preprocessing runs in a separate Ray actor pool
          (decoupled from the training/GPU workers), so the netCDF fork-safety
          issue no longer pins us to num_workers=0. Each training worker consumes
          its shard via ray.train.get_dataset_shard("train").iter_torch_batches.
          DESIGN INTENT: decoupling preprocessing from the GPU workers is meant
          to recover the throughput that num_workers=0 leaves on the table (a
          rough ~2x is the design target, NOT a measured result yet). Both
          --data synthetic and --data era5 are wired: era5 is made fork-safe by
          materializing every item to numpy on the driver before handing the
          rows to the Ray Data actor pool (see _build_ray_dataset).

Perf knobs (mixed precision + gradient accumulation):
  --amp       : OPT-IN bf16 autocast on the forward only (default OFF). When off
                the loop reproduces the exact fp32 baseline numerics. When on it
                only takes effect on CUDA; on CPU it stays a no-op. bf16 shares
                fp32's exponent range, so no GradScaler is needed and the
                optimizer / master weights stay fp32.
  --grad-accum: accumulate this many micro-batches per optimizer step (default 1,
                numerically identical to the original single-step loop). Under
                DDP the non-stepping micro-batches run inside no_sync() so the
                gradient all-reduce only fires on the stepping micro-batch.

Data-parallel correctness (fixed 2026-09-22): the loss must come from the DDP
wrapper, rf(fine, cond, coarse_mass), because DDP arms its all-reduce inside its
own forward. Before the fix the loop called rf.module.training_loss(...), and
each rank trained an independent replica. samples_per_sec is now the all-reduced
sample count over the slowest rank's epoch time. tests/test_train_ray_ddp.py
checks both on 2 gloo CPU ranks.
"""
from __future__ import annotations
import argparse
import contextlib


def _build_dataset(config):
    """Construct the torch Dataset used by either ingest backend.

    Kept as a helper so the synthetic and era5 branches live in exactly one
    place and both ingest backends build an identical dataset object.
    """
    if config["data"] == "era5":
        from pcrf.data import ERA5Dataset
        ds = ERA5Dataset(config["coarse"], config["target"], split="train",
                         fine=config["fine"], sat_bands=config["sat_bands"])
    else:
        from pcrf.data import SyntheticDownscaleDataset
        ds = SyntheticDownscaleDataset(n=config["n"], fine=config["fine"],
                                       factor=4, sat_bands=config["sat_bands"])
    return ds


def _peek_channels(config):
    """Return (in_ch, cond_ch) by materializing a single item once on the driver.

    The Ray Data backend needs the channel counts to build the model on each
    worker, but we deliberately do NOT keep the dataset object around: under
    --ingest ray the rows are produced by the Ray Data actor pool, not here.
    For synthetic this is cheap; for era5 it opens the netCDF once on the driver
    (read-only) purely to read tensor shapes.
    """
    ds = _build_dataset(config)
    item0 = ds[0]
    return int(item0["fine"].shape[0]), int(item0["cond"].shape[0])


def _amp_autocast_ctx(amp_enabled, torch):
    """Return the forward-pass context manager.

    bf16 autocast when amp_enabled, otherwise a null context so the forward runs
    in plain fp32. Factored out so the gating logic is unit-testable without a
    GPU: the default (amp_enabled=False) path is the exact fp32 baseline.
    """
    if amp_enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _resolve_amp_enabled(use_amp, use_gpu, cuda_available):
    """Decide whether bf16 autocast actually runs.

    bf16 autocast is a CUDA-only path here: it is enabled only when the user
    opted in (use_amp), the run is a GPU run (use_gpu), AND CUDA is present.
    On CPU this always returns False, so the fp32 baseline numerics are exact.
    Pure function so it can be unit-tested without torch on a real GPU.
    """
    return bool(use_amp) and bool(use_gpu) and bool(cuda_available)


def _is_stepping_micro(micro_idx, grad_accum):
    """True when this micro-batch (0-based index) triggers an optimizer step.

    A step fires on every grad_accum-th micro-batch. With grad_accum == 1 every
    micro-batch steps (so no_sync is never selected). Pure helper for testing
    the no_sync selection logic.
    """
    return ((micro_idx + 1) % grad_accum) == 0


def _accum_sync_ctx(rf, stepping):
    """Return the DDP comms context for one micro-batch.

    On a NON-stepping micro-batch we want to skip the gradient all-reduce, so we
    enter rf.no_sync() when available (rf is the prepare_model handle / DDP
    wrapper; do NOT call no_sync on core == rf.module). On the stepping
    micro-batch (or when rf has no no_sync attribute, e.g. a bare module on CPU)
    we use a null context so the all-reduce fires normally. With grad_accum == 1
    every micro-batch is stepping, so this always returns the null context and
    the path reduces exactly to the original loop.
    """
    if (not stepping) and hasattr(rf, "no_sync"):
        return rf.no_sync()
    return contextlib.nullcontext()


def _micro_step(rf, batch, stepping, grad_accum, amp_enabled, torch):
    """Run the forward and backward for one micro-batch and return (loss, logs).

    The loss comes from rf(...), the prepare_model handle. When rf is a DDP
    wrapper, DDP.forward arms the gradient all-reduce for the backward that
    follows. Calling rf.module.training_loss(...) skips DDP.forward, so the
    ranks never average gradients and each one trains an independent replica.
    The sweeps recorded before 2026-09-22 ran with that bug. With one worker,
    prepare_model returns the bare RectifiedFlow, and rf(...) calls its
    forward, which is training_loss.

    On a non-stepping micro-batch the forward and backward run inside
    rf.no_sync(), so the all-reduce waits for the stepping micro-batch. Only
    the forward runs under bf16 autocast (a null context when amp is off). The
    loss is scaled by 1/grad_accum so the accumulated gradient is the mean over
    the window. With grad_accum == 1 the scale is 1 and every micro-batch syncs.
    """
    with _accum_sync_ctx(rf, stepping):
        with _amp_autocast_ctx(amp_enabled, torch):
            loss, logs = rf(batch["fine"], batch["cond"], batch["coarse_mass"])
        (loss / grad_accum).backward()
    return loss, logs


def _global_throughput(n_local, dt_local, torch, device):
    """Return (n_global, dt_max, samples_per_sec) for one epoch over all ranks.

    samples_per_sec is the sum of the per-rank sample counts divided by the
    slowest rank's epoch time. The older formula multiplied rank 0's local rate
    by the world size, which assumed every rank was equally fast. When
    torch.distributed is not initialized, or the world size is 1, the local
    values come back unchanged. device must suit the process-group backend
    (CUDA tensors for NCCL, CPU tensors for gloo).
    """
    dist = torch.distributed
    n_global, dt_max = float(n_local), float(dt_local)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        n = torch.tensor([n_global], dtype=torch.float64, device=device)
        t = torch.tensor([dt_max], dtype=torch.float64, device=device)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        n_global, dt_max = float(n.item()), float(t.item())
    return n_global, dt_max, n_global / max(dt_max, 1e-6)


def _with_last(iterable):
    """Yield (item, is_last) for each item of iterable.

    Lets the train loop force a SYNCED backward on the FINAL micro-batch of an
    epoch. Without this, a trailing partial accumulation window (micro-batch
    count not a multiple of grad_accum) would all-reduce-skip its last
    micro-batch (it runs under no_sync), and the flush opt.step() would update on
    un-synced gradients, desyncing the DDP ranks. Pure helper for testing.
    """
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for cur in it:
        yield prev, False
        prev = cur
    yield prev, True


def train_loop_per_worker(config):
    import os
    import time
    import tempfile
    import torch
    from torch.utils.data import DataLoader
    import ray.train
    from ray.train import Checkpoint
    from ray.train.torch import prepare_data_loader, prepare_model

    from pcrf import flow as f
    from pcrf import model as m
    from pcrf import physics as p

    ctx = ray.train.get_context()
    world = ctx.get_world_size()
    rank = ctx.get_world_rank()

    ingest = config.get("ingest", "torch")

    # --- mixed precision + gradient accumulation knobs ------------------------
    # amp: OPT-IN bf16 autocast on the forward only. Default OFF, so the default
    # run reproduces the exact fp32 baseline numerics. bf16 has the same exponent
    # range as fp32, so it needs NO GradScaler and the optimizer / master weights
    # stay fp32. It is only meaningful on CUDA; on CPU there is no bf16 fast path
    # so we force it off regardless of the flag.
    # grad_accum: number of micro-batches to accumulate before each opt.step().
    # The effective (global) batch is batch * grad_accum * world. The loss is
    # scaled by 1/grad_accum so the averaged gradient matches a single large
    # batch. With grad_accum == 1 this reduces EXACTLY to the original loop, so
    # default numeric behavior is unchanged.
    use_amp = bool(config.get("amp", False))
    grad_accum = max(1, int(config.get("grad_accum", 1)))
    amp_enabled = _resolve_amp_enabled(
        use_amp, config.get("gpu", False), torch.cuda.is_available())
    if rank == 0:
        print(f"[cfg] ingest={ingest} amp(bf16)={amp_enabled} "
              f"grad_accum={grad_accum} world={world}")

    # --- dataset / channel discovery -----------------------------------------
    # In the ray-ingest path the rows arrive from get_dataset_shard(), so we do
    # not build a torch Dataset on the worker; we only need the channel counts,
    # which are passed through config by the driver. In the torch path we build
    # the Dataset here exactly as before.
    if ingest == "ray":
        in_ch = int(config["in_ch"])
        cond_ch = int(config["cond_ch"])
        ds = None
    else:
        ds = _build_dataset(config)
        in_ch = ds[0]["fine"].shape[0]
        cond_ch = ds[0]["cond"].shape[0]

    spec = p.ChannelSpec(precip=0, u=1, v=2) if in_ch >= 3 else p.ChannelSpec(precip=0)

    net = m.UNet(in_ch=in_ch, cond_ch=cond_ch, base=config["base"],
                 ch_mults=(1, 2, 4), attn_at=(2,))
    rf = f.RectifiedFlow(net, p.PhysicsLoss(spec))
    rf = prepare_model(rf)  # wraps in DistributedDataParallel across workers
    # core is the bare RectifiedFlow. Use it for parameters and checkpoints
    # only. The loss must come from rf(...) (see _micro_step), never from
    # core.training_loss, or DDP skips the gradient all-reduce.
    core = rf.module if hasattr(rf, "module") else rf
    opt = torch.optim.AdamW(core.parameters(), lr=config["lr"])

    # --- data iteration handle -----------------------------------------------
    # torch backend: a prepared torch DataLoader (DistributedSampler inside).
    # ray   backend: this worker's Ray Data shard. iter_torch_batches() handles
    #                sharding across workers and collation into stacked tensors;
    #                there is no torch DataLoader and no DistributedSampler.
    if ingest == "ray":
        train_shard = ray.train.get_dataset_shard("train")
        dl = None
    else:
        dl = prepare_data_loader(
            DataLoader(ds, batch_size=config["batch"], shuffle=True, drop_last=True))

    # --- resume from the latest checkpoint if Ray restarted the job ----------
    start_epoch = 0
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as cd:
            state = torch.load(os.path.join(cd, "state.pt"), map_location="cpu")
        core.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = int(state["epoch"]) + 1
        if rank == 0:
            print(f"[resume] from epoch {start_epoch}")

    device = ray.train.torch.get_device()

    def iter_batches(epoch):
        """Yield batches as dicts of tensors for either ingest backend.

        Both paths yield dicts with keys: fine, cond, coarse_mass, matching
        rf(b["fine"], b["cond"], b["coarse_mass"]).
        """
        if ingest == "ray":
            # iter_torch_batches stacks same-keyed rows into batched tensors and
            # moves them to this worker's device. local_shuffle_buffer_size gives
            # per-epoch shuffling without a global sort (cheap, streaming).
            for b in train_shard.iter_torch_batches(
                    batch_size=config["batch"],
                    local_shuffle_buffer_size=config.get("shuffle_buffer", 256),
                    drop_last=True,
                    device=device):
                yield b
        else:
            if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(epoch)
            for b in dl:
                yield b

    for epoch in range(start_epoch, config["epochs"]):
        core.train()
        t0, n_local, logs = time.time(), 0, {}
        # micro_idx counts micro-batches inside the current accumulation window.
        micro_idx = 0
        opt.zero_grad(set_to_none=True)
        for b, is_last in _with_last(iter_batches(epoch)):
            # Force a SYNCED backward on the final micro-batch of the epoch so a
            # trailing partial accumulation window does not all-reduce-skip its
            # last micro-batch, which would desync DDP ranks at the flush step.
            stepping = _is_stepping_micro(micro_idx, grad_accum) or is_last
            # The forward goes through rf (the DDP wrapper), so the all-reduce
            # fires on the stepping micro-batch. Non-stepping micro-batches run
            # under rf.no_sync(). The backward runs outside bf16 autocast, and
            # grads accumulate in fp32 master weights, so no GradScaler is needed.
            _loss, logs = _micro_step(rf, b, stepping, grad_accum, amp_enabled, torch)
            micro_idx += 1
            # Count EVERY micro-batch's samples so samples_per_sec stays correct.
            n_local += b["fine"].shape[0]

            if micro_idx == grad_accum:
                opt.step()
                opt.zero_grad(set_to_none=True)
                micro_idx = 0
        # Flush a trailing partial accumulation window (when the micro-batch count
        # is not a multiple of grad_accum). The final micro-batch was forced to run
        # SYNCED above, so its grads are all-reduced before this step (no DDP
        # desync); this just applies the leftover, already-synced grads.
        if micro_idx > 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        dt = max(time.time() - t0, 1e-6)
        sps_local = n_local / dt
        # Global throughput: all-reduced sample count over the slowest rank's
        # epoch time. Every rank must reach this call, because it is a collective.
        n_global, dt_max, sps_global = _global_throughput(n_local, dt, torch, device)
        metrics = {
            "epoch": epoch,
            "samples_per_sec": sps_global,
            "samples_per_sec_per_worker": sps_local,
            "samples_per_epoch": n_global,
            "epoch_time_s": dt_max,              # slowest rank
            "epoch_time_s_per_worker": dt,       # this rank
            "world_size": world,
            **{k: float(v) for k, v in logs.items()},
        }
        # Rank 0 writes the checkpoint; all ranks must call report().
        if rank == 0:
            with tempfile.TemporaryDirectory() as td:
                torch.save({"model": core.state_dict(), "opt": opt.state_dict(),
                            "epoch": epoch, "in_ch": in_ch, "cond_ch": cond_ch},
                           os.path.join(td, "state.pt"))
                ray.train.report(metrics, checkpoint=Checkpoint.from_directory(td))
        else:
            ray.train.report(metrics)


def _item_to_numpy_row(item):
    """Convert one dataset item (torch tensors) into a pure-numpy row.

    Returns a dict keyed fine/cond/coarse_mass whose values are np.float32
    ndarrays, NEVER torch tensors. This is the contract the Ray Data UDF emits:
    a ray.data row crosses the cloudpickle / Arrow boundary into a Ray worker as
    numpy, so the row carries no torch tensor and nothing downstream runs a torch
    op on a tensor produced inside a freshly-unpickled worker. Emitting torch
    tensors here is what triggered the worker-side
    "NameError: name '_C' is not defined" (torch._C not initialized in the
    cloudpickled Ray worker); numpy rows sidestep that entirely.

    coarse_mass is emitted as a 1-element [1] array so it collates to [B,1] and
    PhysicsLoss.mass_penalty reduces it to [B] via .view(-1).

    Pure function (torch-tensor item in -> numpy dict out) so it is unit-testable
    and so the "numpy, not torch" contract can be asserted directly in a test.
    """
    import numpy as np

    def _np32(x):
        # torch tensor -> owned, contiguous float32 numpy array. detach + cpu
        # first, then ascontiguousarray to guarantee an INDEPENDENT buffer (not a
        # view sharing storage with the source tensor), so the emitted row holds
        # no reference back into a torch tensor.
        arr = x.detach().cpu().numpy()
        return np.ascontiguousarray(arr, dtype=np.float32)

    return {
        "fine": _np32(item["fine"]),
        "cond": _np32(item["cond"]),
        # coarse_mass is a scalar torch tensor; float() then wrap as a [1] array
        # so it collates to [B,1] (reduced to [B] by mass_penalty .view(-1)).
        "coarse_mass": np.asarray([float(item["coarse_mass"])], dtype=np.float32),
    }


def _build_ray_dataset(config):
    """Build a ray.data.Dataset whose rows are {"fine","cond","coarse_mass"}.

    Ray 2.40 API. We deliberately do NOT use ray.data.from_torch(ds) directly:
    from_torch wraps each __getitem__ result under an "item" key (rows look like
    {"item": {...}}), which would break the fine/cond/coarse_mass keys that
    iter_torch_batches + rf(...) expect. Instead we drive the torch
    Dataset through a flat_map so every produced row already has the three keys
    as numpy arrays, and Ray's actor pool runs that work off the training
    workers. iter_torch_batches then collates the numpy columns into batched
    torch tensors with the keys preserved.

    WORKER torch._C FIX (corrected root cause): the flat_map UDF is fully
    TORCH-FREE. It generates each synthetic item with pcrf.synth_numpy (numpy
    only) and emits np.float32 rows; it never imports torch. The earlier bug was
    NOT a torch tensor crossing the Arrow boundary -- it was ``import torch``
    ITSELF failing inside the Ray Data worker with
    "NameError: name '_C' is not defined" (torch's compiled _C extension is not
    initialized in a forked/spawned Ray worker). The previous UDF still did
    ``from pcrf.data import SyntheticDownscaleDataset``, and pcrf/data.py imports
    torch at module top, so merely importing the dataset class dragged torch into
    the worker and tripped the _C error. Generating the item with numpy keeps the
    worker process torch-free end to end; iter_torch_batches converts the numpy
    columns to torch tensors back on the GPU training worker, where torch is
    fully imported. Batch dict keys (fine/cond/coarse_mass) and shapes are
    unchanged.

    SYNTHETIC: fully supported and exercised end to end. Each row is one item
    index; an actor pool calls SyntheticDownscaleDataset.__getitem__, which is
    pure-compute and fork-safe.

    ERA5: ERA5Dataset holds live netCDF handles that are not fork-safe, so we
    never hand the dataset object to remote actors. Instead we materialize every
    item to numpy ON THE DRIVER (where torch is imported), via _item_to_numpy_row,
    the same {fine,cond,coarse_mass} numpy contract the synthetic path emits.
    After materialization no live netCDF handle is referenced, so the rows are
    picklable and safe to ship to the Ray Data actor pool.
    """
    if config["data"] == "era5":
        # Fork-safe ERA5 Ray Data source. ERA5Dataset opens netCDF/xarray handles
        # in __init__ and reads them in __getitem__; those handles are NOT
        # fork-safe, so shipping the dataset object to Ray actors would crash a
        # forked/spawned worker. We sidestep that by reading every item on the
        # driver and converting it to a pure-numpy row with _item_to_numpy_row
        # (the same contract the synthetic flat_map emits), then dropping the
        # dataset so no open handle survives into the rows handed to Ray.
        #
        # This is the eager in-memory export (option 1 of the original design
        # note): fine for the single-year spans used here. For very large
        # multi-year spans, switch to the on-disk variant (write per-time Parquet
        # shards and ray.data.read_parquet them) so the driver does not hold the
        # whole split in memory; the row schema is identical either way.
        import ray.data
        from pcrf.data import ERA5Dataset
        ds = ERA5Dataset(config["coarse"], config["target"], split="train",
                         fine=int(config["fine"]), sat_bands=int(config["sat_bands"]))
        rows = [_item_to_numpy_row(ds[i]) for i in range(len(ds))]
        del ds  # drop the open netCDF handles before the rows go to Ray
        return ray.data.from_items(rows)

    # --- synthetic: fork-safe, pure-compute item generation ---------------------
    import ray.data

    n = int(config["n"])
    fine = int(config["fine"])
    sat_bands = int(config["sat_bands"])

    def _gen_item(row):
        # row is {"idx": i}. Build one synthetic item here, inside the Ray Data
        # actor pool, as a PURE-NUMPY row. This UDF imports NO torch:
        # pcrf.synth_numpy is torch-free (numpy only), so the Ray Data worker
        # process never runs `import torch` and the worker-side
        # "NameError: name '_C' is not defined" (torch._C uninitialized in a
        # forked/spawned Ray worker) cannot occur. iter_torch_batches converts
        # the numpy columns to torch tensors back on the GPU training worker,
        # where torch is fully imported. _gen_item is a local closure, so
        # cloudpickle serializes it by value -- the worker does not import this
        # (torch-importing) train_ray module to reconstruct it.
        from pcrf.synth_numpy import synth_item_numpy
        return [synth_item_numpy(int(row["idx"]), fine=fine, factor=4,
                                 sat_bands=sat_bands)]

    base = ray.data.from_items([{"idx": i} for i in range(n)])
    # flat_map with a callable class would let Ray reuse actors; a plain fn is
    # simplest and still runs in Ray's task/actor pool, decoupled from training.
    return base.flat_map(_gen_item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "era5"])
    ap.add_argument("--ingest", default="torch", choices=["torch", "ray"],
                    help="data backend: torch (prepare_data_loader, default) or "
                         "ray (Ray Data actor-pool streaming, decoupled from "
                         "training workers; sidesteps netCDF num_workers=0). "
                         "ray currently supports --data synthetic only.")
    ap.add_argument("--coarse", default="")     # ERA5 coarse NetCDF (or glob)
    ap.add_argument("--target", default="")     # ERA5 high-res precip target
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n", type=int, default=512)        # synthetic size
    ap.add_argument("--fine", type=int, default=128)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sat_bands", type=int, default=4)
    ap.add_argument("--shuffle_buffer", type=int, default=256,
                    help="local_shuffle_buffer_size for --ingest ray (per-worker "
                         "streaming shuffle window); ignored by --ingest torch")
    # bf16 mixed precision: OPT-IN (default OFF). When omitted the loop runs the
    # exact fp32 baseline numerics. When passed it only takes effect on GPU
    # (CUDA); on CPU it is a no-op.
    ap.add_argument("--amp", dest="amp", action="store_true", default=False,
                    help="opt in to bf16 autocast on GPU (default OFF; CPU no-op)")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="explicitly disable bf16 autocast (the default; exact "
                         "fp32 baseline)")
    # Gradient accumulation: accumulate this many micro-batches per opt.step() to
    # raise the effective batch and saturate the small 7.6M UNet. Default 1 keeps
    # numerics identical to the original single-step loop.
    ap.add_argument("--grad-accum", dest="grad_accum", type=int, default=1,
                    help="micro-batches accumulated per optimizer step (default 1)")
    ap.add_argument("--storage", default="./ray_results",
                    help="RunConfig storage_path (local dir, or s3://... on a managed cluster)")
    ap.add_argument("--name", default="pcrf")
    ap.add_argument("--max_failures", type=int, default=3,
                    help="auto-restart-from-checkpoint count for spot/preemption (-1 = unlimited)")
    args = ap.parse_args()

    from ray.train import ScalingConfig, RunConfig, FailureConfig
    from ray.train.torch import TorchTrainer

    train_loop_config = dict(
        data=args.data, ingest=args.ingest, coarse=args.coarse, target=args.target,
        n=args.n, fine=args.fine, base=args.base, batch=args.batch,
        lr=args.lr, epochs=args.epochs, sat_bands=args.sat_bands,
        shuffle_buffer=args.shuffle_buffer,
        gpu=args.gpu, amp=args.amp, grad_accum=args.grad_accum)

    # The ray-ingest backend hands a ray.data.Dataset to the trainer and resolves
    # the model channel counts once on the driver (no torch Dataset on workers).
    datasets = None
    if args.ingest == "ray":
        in_ch, cond_ch = _peek_channels(train_loop_config)
        train_loop_config["in_ch"] = in_ch
        train_loop_config["cond_ch"] = cond_ch
        datasets = {"train": _build_ray_dataset(train_loop_config)}

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=train_loop_config,
        datasets=datasets,
        scaling_config=ScalingConfig(
            num_workers=args.workers, use_gpu=args.gpu,
            resources_per_worker={"GPU": 1} if args.gpu else None),
        run_config=RunConfig(
            storage_path=args.storage, name=args.name,
            # Survive spot/preemption: Ray restarts the run and the worker resumes
            # from the latest checkpoint via ray.train.get_checkpoint() above.
            failure_config=FailureConfig(max_failures=args.max_failures)),
    )
    result = trainer.fit()
    print("[ray-train] done:", result.metrics)


if __name__ == "__main__":
    main()
