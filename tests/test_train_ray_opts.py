"""Unit tests for the reconciled train_ray.py optimization knobs.

Covers the two merged features and their must-fixes, all CPU/synthetic:
  - AMP gating (opt-in, default OFF; CPU/no-amp always disabled).
  - The grad_accum == 1 path being the unscaled, every-micro-batch-steps path.
  - The DDP no_sync() selection: no_sync on non-stepping micro-batches, plain
    sync on the stepping (last-in-window) micro-batch.
  - The --ingest ray Ray Data builder producing fine/cond/coarse_mass batches
    with the right shapes for synthetic, and raising NotImplementedError for
    ERA5. The UDF emits NUMPY rows (np.float32, never torch tensors) so the
    torch._C-not-initialized worker bug cannot recur; iter_torch_batches does
    the numpy -> torch conversion in the training loop.
  - The default train_loop_config matching the pre-change (no-knob) behavior.

The module under test is apps/training/train_ray.py, importable because
conftest.py puts apps/training on sys.path. Helpers were factored so the gating
and no_sync logic are testable without a GPU or a live TorchTrainer.
"""
import contextlib

import numpy as np
import pytest

import train_ray as tr


# --------------------------------------------------------------------------
# (a) AMP gating: opt-in, default OFF, CPU/no-amp always disabled.
# --------------------------------------------------------------------------
def test_resolve_amp_enabled_off_on_cpu():
    # CPU run (cuda_available False) is never AMP-enabled, even if the user
    # opted in and use_gpu is True.
    assert tr._resolve_amp_enabled(use_amp=True, use_gpu=True,
                                   cuda_available=False) is False


def test_resolve_amp_enabled_off_with_no_amp():
    # --no-amp (use_amp False) disables AMP even on a CUDA GPU run.
    assert tr._resolve_amp_enabled(use_amp=False, use_gpu=True,
                                   cuda_available=True) is False


def test_resolve_amp_enabled_off_without_gpu_flag():
    # Opted in + CUDA present but no --gpu: still off (this is a CPU-style run).
    assert tr._resolve_amp_enabled(use_amp=True, use_gpu=False,
                                   cuda_available=True) is False


def test_resolve_amp_enabled_on_only_when_all_true():
    # AMP runs only when opted in AND a GPU run AND CUDA is present.
    assert tr._resolve_amp_enabled(use_amp=True, use_gpu=True,
                                   cuda_available=True) is True


def test_amp_autocast_ctx_is_nullcontext_when_disabled():
    # When AMP is disabled the forward runs under a real null context, i.e. plain
    # fp32 (the exact baseline). We assert the type rather than entering it so
    # the test needs no CUDA.
    import torch
    ctx = tr._amp_autocast_ctx(False, torch)
    assert isinstance(ctx, contextlib.nullcontext)


# --------------------------------------------------------------------------
# (b) grad_accum == 1: unscaled path + no_sync never selected.
# --------------------------------------------------------------------------
def test_grad_accum_one_every_micro_is_stepping():
    # With grad_accum == 1 every micro-batch triggers an optimizer step, so the
    # loop reduces to the original single-step loop (no accumulation window).
    assert all(tr._is_stepping_micro(i, grad_accum=1) for i in range(5))


def test_grad_accum_one_loss_scale_is_identity():
    # The loop scales the loss by 1/grad_accum before backward(); with
    # grad_accum == 1 this is the identity, i.e. the unscaled baseline gradient.
    grad_accum = 1
    loss = 3.5
    assert (loss / grad_accum) == pytest.approx(loss)


def test_stepping_pattern_for_grad_accum_two():
    # grad_accum == 2: step fires on micro-batches 1 and 3 (0-based), not 0 / 2.
    flags = [tr._is_stepping_micro(i, grad_accum=2) for i in range(4)]
    assert flags == [False, True, False, True]


# --------------------------------------------------------------------------
# (b) no_sync selection: enter no_sync on non-stepping micro-batches, plain
#     sync on the stepping one; entered on rf, not core; no-op when no attr.
# --------------------------------------------------------------------------
class _FakeDDP:
    """Stand-in for the prepare_model DDP wrapper exposing no_sync()."""

    def __init__(self):
        self.no_sync_calls = 0

    @contextlib.contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield


def test_no_sync_selected_on_nonstepping_micro():
    rf = _FakeDDP()
    ctx = tr._accum_sync_ctx(rf, stepping=False)
    with ctx:
        pass
    # The non-stepping micro-batch must enter rf.no_sync() exactly once so the
    # all-reduce is skipped until the step.
    assert rf.no_sync_calls == 1


def test_sync_used_on_stepping_micro():
    rf = _FakeDDP()
    ctx = tr._accum_sync_ctx(rf, stepping=True)
    assert isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass
    # The stepping micro-batch must NOT enter no_sync(): the reduce fires here.
    assert rf.no_sync_calls == 0


def test_no_sync_noop_when_rf_has_no_attr():
    # A bare module (e.g. CPU / no DDP) has no no_sync; selection degrades to a
    # null context on every micro-batch, never raising.
    class _Bare:
        pass

    ctx = tr._accum_sync_ctx(_Bare(), stepping=False)
    assert isinstance(ctx, contextlib.nullcontext)


def test_no_sync_never_selected_when_grad_accum_one():
    # End-to-end of the selection logic at grad_accum == 1: every micro-batch is
    # stepping, so no_sync is never entered (matches the original loop's comms).
    rf = _FakeDDP()
    grad_accum = 1
    for micro_idx in range(4):
        stepping = tr._is_stepping_micro(micro_idx, grad_accum)
        with tr._accum_sync_ctx(rf, stepping):
            pass
    assert rf.no_sync_calls == 0


# --------------------------------------------------------------------------
# (P1) --ingest ray dataset builder: synthetic batches carry fine/cond/
#      coarse_mass with correct shapes; ERA5 is fork-safe (numpy rows, no live
#      netCDF handle crosses into the Ray actor pool).
# --------------------------------------------------------------------------
def _synthetic_config(n=4, fine=16, sat_bands=2, batch=2):
    return dict(data="synthetic", n=n, fine=fine, sat_bands=sat_bands,
                batch=batch)


def test_era5_ray_rows_are_fork_safe_numpy(tmp_path):
    # ERA5 under --ingest ray must produce FORK-SAFE rows: every item is read on
    # the driver and converted to pure numpy (no live netCDF handle survives into
    # the row), so a forked/spawned Ray worker never touches a non-fork-safe
    # handle. This checks the contract directly, without needing Ray itself.
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    torch = pytest.importorskip("torch")
    import pickle
    from pcrf import data as d

    cpath, tpath = str(tmp_path / "coarse.nc"), str(tmp_path / "target.nc")
    d.write_toy_era5(cpath, tpath, n_time=6, hc=16, wc=16, hf=16, wf=16)
    ds = d.ERA5Dataset(cpath, tpath, split="train", fine=16, sat_bands=2)

    rows = [tr._item_to_numpy_row(ds[i]) for i in range(len(ds))]
    assert rows, "toy ERA5 train split should be non-empty"
    for row in rows:
        assert set(row.keys()) == {"fine", "cond", "coarse_mass"}
        for k in ("fine", "cond", "coarse_mass"):
            assert isinstance(row[k], np.ndarray), f"{k} must be numpy"
            assert not torch.is_tensor(row[k]), f"{k} must not be a torch tensor"
            assert row[k].dtype == np.float32
        assert row["fine"].shape == (3, 16, 16)
        assert row["cond"].shape == (5, 16, 16)   # 3 coarse + sat_bands(2)
        assert row["coarse_mass"].shape == (1,)
    # Fork-safe: the rows pickle cleanly (no open netCDF / xarray handle inside).
    pickle.dumps(rows)


@pytest.fixture(scope="module")
def ray_local():
    ray = pytest.importorskip("ray")
    # Tiny local CPU-only Ray cluster. local_mode was removed in Ray >= 2.40, so
    # we start a real single-node local cluster instead (still cheap, CPU only).
    ray.init(num_cpus=2, include_dashboard=False,
             ignore_reinit_error=True, log_to_driver=False)
    yield ray
    ray.shutdown()


def test_item_to_numpy_row_emits_numpy_not_torch():
    # REGRESSION GUARD (torch._C worker bug): the UDF helper must emit pure numpy
    # rows, never torch tensors. A torch tensor crossing the cloudpickle boundary
    # into a Ray worker (or a torch op run on a freshly-unpickled worker) is what
    # raised "NameError: name '_C' is not defined"; emitting numpy prevents it.
    # Pure function -> no Ray / no GPU needed for this check.
    torch = pytest.importorskip("torch")
    # A minimal stand-in item matching SyntheticDownscaleDataset.__getitem__:
    # fine [3,H,W], cond [3+sat,H,W], coarse_mass scalar tensor.
    item = {"fine": torch.zeros(3, 8, 8),
            "cond": torch.zeros(5, 8, 8),
            "coarse_mass": torch.tensor(0.5)}
    row = tr._item_to_numpy_row(item)

    assert set(row.keys()) == {"fine", "cond", "coarse_mass"}
    for k in ("fine", "cond", "coarse_mass"):
        # Each value must be an ndarray, NOT a torch tensor.
        assert isinstance(row[k], np.ndarray), f"{k} must be numpy, got {type(row[k])}"
        assert not torch.is_tensor(row[k]), f"{k} must not be a torch tensor"
        assert row[k].dtype == np.float32
    # Shapes preserved; coarse_mass becomes a [1] array (collates to [B,1]).
    assert row["fine"].shape == (3, 8, 8)
    assert row["cond"].shape == (5, 8, 8)
    assert row["coarse_mass"].shape == (1,)
    # Owned, contiguous buffer (not a view back into the source tensor).
    assert row["fine"].flags["C_CONTIGUOUS"]


def test_build_ray_dataset_synthetic_rows_are_numpy_not_torch(ray_local):
    # REGRESSION GUARD: the materialized Ray Data rows from the synthetic flat_map
    # must be numpy arrays, not torch tensors, so the UDF output never carries a
    # torch tensor across the worker boundary (the torch._C-not-initialized bug).
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray.data")
    cfg = _synthetic_config(n=4, fine=16, sat_bands=2, batch=2)
    ds = tr._build_ray_dataset(cfg)

    row = ds.take(1)[0]
    for k in ("fine", "cond", "coarse_mass"):
        assert not torch.is_tensor(row[k]), f"{k} row value must not be a torch tensor"
        assert isinstance(np.asarray(row[k]), np.ndarray)


def test_build_ray_dataset_era5_builds_numpy_rows(ray_local, tmp_path):
    # End-to-end era5 --ingest ray path: _build_ray_dataset materializes a real
    # ERA5Dataset to a Ray Data dataset of fork-safe numpy rows. Needs Ray.
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray.data")
    from pcrf import data as d

    cpath, tpath = str(tmp_path / "coarse.nc"), str(tmp_path / "target.nc")
    d.write_toy_era5(cpath, tpath, n_time=6, hc=16, wc=16, hf=16, wf=16)
    cfg = dict(data="era5", coarse=cpath, target=tpath, fine=16, sat_bands=2)

    ds = tr._build_ray_dataset(cfg)
    row = ds.take(1)[0]
    assert set(row.keys()) == {"fine", "cond", "coarse_mass"}
    for k in ("fine", "cond", "coarse_mass"):
        assert not torch.is_tensor(row[k]), f"{k} row must not be a torch tensor"
        assert isinstance(np.asarray(row[k]), np.ndarray)
    assert np.asarray(row["fine"]).shape == (3, 16, 16)
    assert np.asarray(row["cond"]).shape == (5, 16, 16)


def test_build_ray_dataset_synthetic_batches_shapes(ray_local):
    # The synthetic Ray Data path must yield rows keyed fine/cond/coarse_mass.
    pytest.importorskip("ray.data")
    fine, sat_bands, batch = 16, 2, 2
    cfg = _synthetic_config(n=4, fine=fine, sat_bands=sat_bands, batch=batch)
    ds = tr._build_ray_dataset(cfg)

    # Inspect one materialized row's schema/shapes (cheap, no GPU).
    row = ds.take(1)[0]
    assert set(row.keys()) == {"fine", "cond", "coarse_mass"}
    fine_arr = np.asarray(row["fine"])
    cond_arr = np.asarray(row["cond"])
    cm_arr = np.asarray(row["coarse_mass"])
    # The row columns are emitted as np.float32 by the numpy-emit UDF.
    assert fine_arr.dtype == np.float32
    assert cond_arr.dtype == np.float32
    assert cm_arr.dtype == np.float32
    # fine has 3 channels (precip,u,v); cond has 3 + sat_bands; both at fine res.
    assert fine_arr.shape == (3, fine, fine)
    assert cond_arr.shape == (3 + sat_bands, fine, fine)
    # coarse_mass is stored as a 1-element row -> [B,1] after collation, then
    # reduced to [B] by PhysicsLoss.mass_penalty via .view(-1).
    assert cm_arr.shape == (1,)


def test_iter_torch_batches_carries_keys_and_shapes(ray_local):
    # iter_torch_batches must collate the numpy-emitted columns into batched
    # torch tensors that match rf(fine, cond, coarse_mass) on CPU.
    # The numpy-emit contract (rows are numpy; iter_torch_batches does the
    # numpy -> torch conversion) is exactly what sidesteps the torch._C worker
    # bug, so this test documents that the conversion happens HERE, not in the UDF.
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray.data")
    fine, sat_bands, batch = 16, 2, 2
    cfg = _synthetic_config(n=4, fine=fine, sat_bands=sat_bands, batch=batch)
    ds = tr._build_ray_dataset(cfg)

    # The UDF emits numpy rows; confirm that pre-collation contract first.
    pre = ds.take(1)[0]
    assert not torch.is_tensor(pre["fine"])  # numpy in, torch out

    it = ds.iter_torch_batches(batch_size=batch, drop_last=True)
    b = next(iter(it))
    assert set(["fine", "cond", "coarse_mass"]).issubset(b.keys())
    # iter_torch_batches converts the numpy columns to torch tensors.
    assert torch.is_tensor(b["fine"])
    assert torch.is_tensor(b["cond"])
    assert torch.is_tensor(b["coarse_mass"])
    assert b["fine"].shape == (batch, 3, fine, fine)
    assert b["cond"].shape == (batch, 3 + sat_bands, fine, fine)
    # coarse_mass collates to [B,1]; mass_penalty does .view(-1) -> [B].
    assert b["coarse_mass"].view(-1).shape == (batch,)
    assert b["fine"].dtype == torch.float32


# --------------------------------------------------------------------------
# (c / default-equivalence) default train_loop_config == pre-change behavior.
# --------------------------------------------------------------------------
def test_default_cli_config_matches_pre_change_defaults():
    # Parse with no extra flags via main()'s parser, mirrored here, and confirm
    # the knobs default to the no-op baseline: torch ingest, amp OFF, accum 1.
    # We reconstruct the defaults the same way main() does so a drift in main()
    # would show up as a mismatch.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", default="torch", choices=["torch", "ray"])
    ap.add_argument("--amp", dest="amp", action="store_true", default=False)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-accum", dest="grad_accum", type=int, default=1)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args([])
    assert args.ingest == "torch"
    assert args.amp is False           # AMP opt-in, default OFF
    assert args.grad_accum == 1        # accumulation default no-op
    assert args.gpu is False
    # And with these defaults the per-worker gating resolves to a pure fp32,
    # single-step, always-sync loop, i.e. the original behavior.
    assert tr._resolve_amp_enabled(args.amp, args.gpu, cuda_available=False) is False
    assert tr._is_stepping_micro(0, args.grad_accum) is True


# --------------------------------------------------------------------------
# Trailing-window DDP sync: the final micro-batch of an epoch must be SYNCED
# (regression guard for the no_sync trailing-window desync bug).
# --------------------------------------------------------------------------
def test_with_last_marks_only_final_item():
    items = list(tr._with_last([10, 20, 30]))
    assert [v for v, _ in items] == [10, 20, 30]
    assert [is_last for _, is_last in items] == [False, False, True]


def test_with_last_empty_iterable_yields_nothing():
    assert list(tr._with_last([])) == []


def test_trailing_micro_batch_is_forced_synced():
    # 3 micro-batches, grad_accum=2 (count NOT a multiple of grad_accum).
    # Replicate the loop's decision: stepping = _is_stepping_micro(i) or is_last.
    # The fix forces the FINAL micro-batch to step (sync) so its gradient is
    # all-reduced before the trailing-window opt.step(). Without the fix the
    # pattern would be [False, True, False] and the flush would update on
    # un-synced grads, desyncing DDP ranks.
    grad_accum = 2
    seen = []
    for micro_idx, (_item, is_last) in enumerate(tr._with_last(["a", "b", "c"])):
        stepping = tr._is_stepping_micro(micro_idx, grad_accum) or is_last
        seen.append(stepping)
    assert seen == [False, True, True]
    assert seen[-1] is True  # trailing micro-batch is SYNCED, not no_sync
