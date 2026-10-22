"""Multi-process DDP tests for apps/training/train_ray.py (CPU, gloo, 2 ranks).

test_train_ray_opts.py checks the no_sync selection with a _FakeDDP stand-in.
A fake cannot check the property that matters for data parallelism, which is
that every rank ends a backward with the same gradient. These tests start two
real processes, join them with torch.distributed (gloo, file rendezvous), wrap
a real RectifiedFlow in DistributedDataParallel, and give each rank different
data. Without a working all-reduce the two ranks' gradients must differ.

The one-step test also runs the pre-fix path (rf.module.training_loss) as a
negative control. It asserts that this path leaves the ranks' gradients
different, which shows the test can detect the bug fixed on 2026-09-22.

Each worker saves its results to tmp_path, and the parent process makes the
assertions, so a failure reports a normal pytest diff.
"""
import copy
import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import train_ray as tr

pytestmark = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="needs torch.distributed with the gloo backend")

WORLD = 2
FINE = 16
TIMEOUT_S = 120


# --------------------------------------------------------------------------
# Helpers that run inside the spawned worker processes. They are module-level
# functions because the spawn start method pickles them by name.
# --------------------------------------------------------------------------
def _build_rf():
    from pcrf import flow as f
    from pcrf import model as m
    from pcrf import physics as p
    # The same seed gives every rank the same initial weights. DDP also
    # broadcasts rank 0's weights when it wraps the module.
    torch.manual_seed(0)
    spec = p.ChannelSpec(precip=0, u=1, v=2)
    net = m.UNet(in_ch=3, cond_ch=5, base=8, ch_mults=(1, 2), attn_at=())
    return f.RectifiedFlow(net, p.PhysicsLoss(spec))


def _batch(rank, micro):
    # Each (rank, micro-batch) pair gets its own data, so ranks never agree
    # by accident.
    g = torch.Generator().manual_seed(1000 * (rank + 1) + micro)
    return {"fine": torch.randn(2, 3, FINE, FINE, generator=g),
            "cond": torch.randn(2, 5, FINE, FINE, generator=g),
            "coarse_mass": torch.rand(2, 1, generator=g)}


def _seed_noise(rank, micro):
    # training_loss draws t and x0 from the global RNG. Reseeding before each
    # call lets a second path replay exactly the same noise.
    torch.manual_seed(7 + 100 * rank + micro)


def _flat_grads(module):
    return torch.cat([q.grad.detach().flatten().clone()
                      for q in module.parameters()])


def _init(rank, store):
    dist.init_process_group("gloo", init_method=f"file://{store}",
                            rank=rank, world_size=WORLD)


def _worker_one_step(rank, store, out_dir):
    _init(rank, store)
    try:
        ddp = DDP(_build_rf())
        b = _batch(rank, 0)

        # Fixed path: the loop's own helper, which calls the DDP wrapper.
        _seed_noise(rank, 0)
        tr._micro_step(ddp, b, stepping=True, grad_accum=1,
                       amp_enabled=False, torch=torch)
        fixed = _flat_grads(ddp)
        ddp.zero_grad(set_to_none=True)

        # Negative control: the pre-fix call, which bypasses DDP.forward.
        _seed_noise(rank, 0)
        loss, _ = ddp.module.training_loss(b["fine"], b["cond"], b["coarse_mass"])
        loss.backward()
        bypass = _flat_grads(ddp)

        torch.save({"fixed": fixed, "bypass": bypass},
                   os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _worker_accum(rank, store, out_dir):
    _init(rank, store)
    try:
        rf = _build_rf()
        ref = copy.deepcopy(rf)   # a plain replica with no DDP, for the reference
        ddp = DDP(rf)
        grad_accum = 2

        after_first = None
        for micro in range(grad_accum):
            stepping = tr._is_stepping_micro(micro, grad_accum)
            _seed_noise(rank, micro)
            tr._micro_step(ddp, _batch(rank, micro), stepping, grad_accum,
                           amp_enabled=False, torch=torch)
            if micro == 0:
                after_first = _flat_grads(ddp)
        synced = _flat_grads(ddp)

        # Reference: replay the same micro-batches and noise on the plain
        # replica, then average the local sums over ranks by hand.
        for micro in range(grad_accum):
            _seed_noise(rank, micro)
            b = _batch(rank, micro)
            loss, _ = ref(b["fine"], b["cond"], b["coarse_mass"])
            (loss / grad_accum).backward()
        expected = _flat_grads(ref)
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)
        expected /= WORLD

        torch.save({"after_first": after_first, "synced": synced,
                    "expected": expected},
                   os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _worker_throughput(rank, store, out_dir):
    _init(rank, store)
    try:
        # Rank 0 is fast with few samples and rank 1 is slow with many.
        n_local, dt_local = [(10, 2.0), (30, 5.0)][rank]
        out = tr._global_throughput(n_local, dt_local, torch, torch.device("cpu"))
        torch.save({"out": torch.tensor(out, dtype=torch.float64)},
                   os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------
# Parent-side driver.
# --------------------------------------------------------------------------
def _run(fn, tmp_path):
    """Spawn WORLD ranks of fn, wait with a deadline, and load each rank's file."""
    store = tmp_path / "rendezvous"
    ctx = mp.start_processes(fn, args=(str(store), str(tmp_path)),
                             nprocs=WORLD, join=False, start_method="spawn")
    deadline = time.monotonic() + TIMEOUT_S
    # join() re-raises a worker's exception in the parent.
    while not ctx.join(timeout=5):
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                proc.terminate()
            pytest.fail(f"DDP workers did not finish within {TIMEOUT_S} s")
    return [torch.load(tmp_path / f"rank{r}.pt") for r in range(WORLD)]


def test_ddp_grads_identical_across_ranks_after_one_backward(tmp_path):
    r0, r1 = _run(_worker_one_step, tmp_path)

    # Fixed path: DDP all-reduced the gradients, so both ranks hold the same bits.
    assert torch.equal(r0["fixed"], r1["fixed"])

    # Negative control: the pre-fix path leaves each rank with its own gradient.
    gap = float((r0["bypass"] - r1["bypass"]).abs().max())
    assert gap > 1e-3, f"bypass grads should differ across ranks, max gap {gap}"

    # The synced gradient is the mean of the two local gradients.
    torch.testing.assert_close(r0["fixed"], (r0["bypass"] + r1["bypass"]) / 2,
                               rtol=1e-5, atol=1e-6)


def test_grad_accum_no_sync_defers_then_syncs(tmp_path):
    r0, r1 = _run(_worker_accum, tmp_path)

    # After the first micro-batch (run under no_sync) the gradients are still
    # local, so they differ across ranks.
    gap = float((r0["after_first"] - r1["after_first"]).abs().max())
    assert gap > 1e-3, f"no_sync micro-batch should stay local, max gap {gap}"

    # After the stepping micro-batch, both ranks hold the same gradient.
    assert torch.equal(r0["synced"], r1["synced"])

    # That gradient equals the mean over ranks of each rank's accumulated,
    # 1/grad_accum-scaled local gradient.
    torch.testing.assert_close(r0["synced"], r0["expected"], rtol=1e-5, atol=1e-6)


def test_global_throughput_sums_counts_over_slowest_rank(tmp_path):
    r0, r1 = _run(_worker_throughput, tmp_path)
    # 10 + 30 samples over the slowest rank's 5.0 s gives 8.0 samples/s. The old
    # formula (rank 0's 10 / 2.0 = 5.0, times 2 ranks) would report 10.0.
    expected = torch.tensor([40.0, 5.0, 8.0], dtype=torch.float64)
    assert torch.equal(r0["out"], expected)
    assert torch.equal(r1["out"], expected)


# --------------------------------------------------------------------------
# Single-process checks: the one-worker path, where prepare_model returns the
# bare RectifiedFlow and torch.distributed may not be initialized.
# --------------------------------------------------------------------------
def test_global_throughput_single_process_is_local():
    assert not dist.is_initialized()
    assert tr._global_throughput(12, 3.0, torch, torch.device("cpu")) == (12.0, 3.0, 4.0)


def test_micro_step_on_bare_module_scales_loss():
    # A bare RectifiedFlow has no no_sync, so a non-stepping micro-batch must
    # still run, with the loss scaled by 1/grad_accum.
    b = _batch(0, 0)
    rf_a, rf_b = _build_rf(), _build_rf()

    _seed_noise(0, 0)
    tr._micro_step(rf_a, b, stepping=False, grad_accum=2,
                   amp_enabled=False, torch=torch)
    _seed_noise(0, 0)
    loss, _ = rf_b.training_loss(b["fine"], b["cond"], b["coarse_mass"])
    loss.backward()

    torch.testing.assert_close(_flat_grads(rf_a), _flat_grads(rf_b) / 2,
                               rtol=1e-5, atol=1e-7)
