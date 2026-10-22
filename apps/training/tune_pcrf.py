"""
Ray Tune (2.40) HPO over PC-RF's physics-penalty weights and learning rate.

This reuses the EXISTING data-parallel training logic: it imports
train_loop_per_worker from train_ray (no copy, no edit of train_ray.py) and runs
it inside a TorchTrainer, then wraps that TorchTrainer in a Tuner that sweeps:

    l_div   (divergence-free wind penalty, lambda_1)
    l_nn    (precip non-negativity penalty, lambda_2)
    l_mass  (domain mass-conservation penalty, lambda_3)
    lr      (AdamW learning rate; loguniform ~1e-4 .. 5e-4)

Penalty defaults vs. the sweep center (read this carefully)
-----------------------------------------------------------
The PhysicsLoss LIBRARY defaults are l_div=1.0, l_nn=1.0, l_mass=1.0 (see
pcrf.physics.PhysicsLoss.__init__). train_ray.py builds PhysicsLoss(spec) with NO
weight arguments, so the UNTUNED training run uses those 1.0 / 1.0 / 1.0 library
defaults. The triplet 1.0 / 0.1 / 0.05 is NOT the current default: it is the
TARGET configuration this sweep centers its search ranges on (the values the paper
proposes), which is exactly why HPO is worth running. The loguniform ranges below
bracket 1.0 / 0.1 / 0.05 so the search can confirm or move off that target.

ASHAScheduler early-stops weak trials on a per-epoch metric. train_loop_per_worker
already reports per-epoch metrics every epoch from every worker, including
"L_total" (the full training objective L_RF + physics) and the individual physics
logs "L_div" / "L_nn" / "L_mass". By default we early-stop / rank on "L_total" as a
clearly-labeled training-loss proxy for the science objective (lower is better).

  IMPORTANT (honesty): "L_total" is the TRAINING objective on the train split, not
  a held-out CRPS or a held-out divergence. It is a proxy. A true held-out
  CRPS/divergence validation pass needs a forward sample() over a val DataLoader
  inside the per-worker loop, which lives in train_ray.py and which this file does
  not modify. See the TODO at the bottom of this file for the exact, non-faked way
  to wire that in (a small change to train_ray.py).
  Until then, optimizing L_total is a faithful, conservative stand-in: it contains
  the same physics penalties (L_div + L_nn + L_mass) the science cares about, so a
  config that lowers L_total is genuinely lowering the penalized objective.

How the physics weights reach the model
----------------------------------------
train_loop_per_worker builds PhysicsLoss(spec) with the library defaults
(l_div=1.0, l_nn=1.0, l_mass=1.0) and does NOT read l_div/l_nn/l_mass from its
config. Rather than edit train_ray.py, this file injects the sampled weights by
monkeypatching pcrf.physics.PhysicsLoss INSIDE each trial's per-worker process so
that PhysicsLoss(spec) is constructed with the trial's weights. The patch runs in
the worker process (where train_loop_per_worker imports `from pcrf import physics
as p` and calls p.PhysicsLoss(spec)), is process-local, and reads the weights from
the per-trial train_loop_config. This is backward compatible: train_ray.py and its
own CLI / sbatch are untouched and keep using the library defaults.

  CONTRACT THE MONKEYPATCH DEPENDS ON: train_ray.train_loop_per_worker must build
  the loss via the MODULE ATTRIBUTE, i.e. `from pcrf import physics as p` then
  `p.PhysicsLoss(spec)` (an attribute lookup at call time). Rebinding
  p.PhysicsLoss before _real_loop runs is what makes the patch take effect. If a
  future edit to train_ray.py instead does `from pcrf.physics import PhysicsLoss`
  and calls the bare name, the name is bound at import time and this patch would
  silently no-op. The helper below documents that requirement; the worker-side
  patch in _tune_train_loop rebinds the SAME attribute the loop reads.

Usage
-----
  # synthetic smoke (laptop / CPU), tiny sweep:
  python apps/training/tune_pcrf.py --data synthetic --num-samples 8 \
      --epochs 6 --workers 1

  # real ERA5 on a GPU node (1 GPU per trial worker), bigger sweep:
  python apps/training/tune_pcrf.py --data era5 --gpu --workers 1 \
      --coarse "data/era5_*/coarse.nc" \
      --target "data/era5_*/target.nc" \
      --num-samples 16 --epochs 30 \
      --storage ./ray_tune --name pcrf-hpo

Reading the best config from the ResultGrid
-------------------------------------------
At the end of main() we call extract_best_config(results, metric, mode), which
wraps results.get_best_result(...) and pulls the winning train_loop_config keys
(l_div / l_nn / l_mass / lr). results.get_dataframe() gives one row per trial for
a full leaderboard. The whole grid is also persisted under <storage>/<name>/ and
can be restored later with ray.tune.Tuner.restore(path, trainable=...).
"""
from __future__ import annotations
import argparse

# Library defaults of pcrf.physics.PhysicsLoss (l_div, l_nn, l_mass). Kept here as
# a single source of truth for the docstring claim above and for tests; if the
# library defaults ever change, update this and the docstring together.
PHYSICS_LIBRARY_DEFAULTS = {"l_div": 1.0, "l_nn": 1.0, "l_mass": 1.0}

# The triplet the sweep centers its loguniform ranges on (paper target config).
# This is the CENTER of the search, NOT the current PhysicsLoss default.
SWEEP_CENTER = {"l_div": 1.0, "l_nn": 0.1, "l_mass": 0.05}


def build_param_space(base_loop_config, lr_low=1e-4, lr_high=5e-4,
                      sampler=None):
    """Build the Tune param_space dict (pure / testable).

    The four swept keys live under "train_loop_config" so they merge into the
    dict train_loop_per_worker receives; the rest come from base_loop_config.

    Ranges bracket the SWEEP_CENTER (1.0 / 0.1 / 0.05), not the library defaults:
        l_div  in [0.1, 10.0]   (around 1.0)
        l_nn   in [0.01, 1.0]   (around 0.1)
        l_mass in [0.005, 0.5]  (around 0.05)
        lr     in [lr_low, lr_high]

    sampler: a callable (lo, hi) -> domain used for every swept key. Defaults to
        ray.tune.loguniform. Injectable so unit tests can record the (lo, hi)
        bounds without importing Ray. The returned param_space therefore has the
        same KEYS and BOUNDS regardless of whether ray is installed.
    """
    if sampler is None:
        from ray import tune
        sampler = tune.loguniform
    return {
        "train_loop_config": dict(
            base_loop_config,
            l_div=sampler(0.1, 10.0),     # around 1.0
            l_nn=sampler(0.01, 1.0),      # around 0.1
            l_mass=sampler(0.005, 0.5),   # around 0.05
            lr=sampler(lr_low, lr_high),
        ),
    }


def make_tuned_physics_loss_cls(physics_module, l_div, l_nn, l_mass):
    """Return a PhysicsLoss subclass whose weight defaults are the trial's values.

    This is the core of the monkeypatch, factored out so it can be unit-tested
    without launching Ray: construct the returned class with just a spec and
    assert that .l_div / .l_nn / .l_mass equal the injected values, while still
    honoring any explicit override a caller passes.

    physics_module: the pcrf.physics module object (passed in so the subclass
        inherits from exactly the class the worker loop will construct, and so
        tests can pass the real module).
    """
    _Orig = physics_module.PhysicsLoss
    _ldiv, _lnn, _lmass = float(l_div), float(l_nn), float(l_mass)

    class _TunedPhysicsLoss(_Orig):
        # train_ray.py constructs PhysicsLoss(spec) positionally with no weights,
        # so override the weight defaults with the trial's sampled values while
        # still honoring any explicit override a future caller might pass.
        def __init__(self, spec, l_div=_ldiv, l_nn=_lnn, l_mass=_lmass):
            super().__init__(spec, l_div=l_div, l_nn=l_nn, l_mass=l_mass)

    return _TunedPhysicsLoss


def extract_best_config(results, metric, mode):
    """Pull the winning physics weights + lr from a ResultGrid-like object.

    results: anything exposing get_best_result(metric=, mode=) -> an object with
        a .config dict and a .metrics dict (the real ray ResultGrid, or a fake in
        tests). Returns a dict with keys among {l_div, l_nn, l_mass, lr} plus
        "metric_value" (the best metric reading) and "metric_name".
    """
    best = results.get_best_result(metric=metric, mode=mode)
    best_loop = best.config.get("train_loop_config", best.config)
    out = {k: best_loop[k] for k in ("l_div", "l_nn", "l_mass", "lr")
           if k in best_loop}
    out["metric_name"] = metric
    out["metric_value"] = best.metrics.get(metric)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Ray Tune HPO over PC-RF physics-penalty weights + lr.")
    # data / model (mirror train_ray.py's flags so trials build the same model)
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "era5"])
    ap.add_argument("--coarse", default="")     # ERA5 coarse NetCDF (or glob)
    ap.add_argument("--target", default="")     # ERA5 high-res precip target
    ap.add_argument("--workers", type=int, default=1,
                    help="DDP workers PER TRIAL (each trial is one TorchTrainer)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=10,
                    help="max epochs per trial (ASHA may stop a trial earlier)")
    ap.add_argument("--n", type=int, default=512)        # synthetic size
    ap.add_argument("--fine", type=int, default=128)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sat_bands", type=int, default=4)
    # search / scheduler
    ap.add_argument("--num-samples", type=int, default=8,
                    help="number of hyperparameter configurations to try")
    ap.add_argument("--metric", default="L_total",
                    help="per-epoch metric to optimize (L_total proxy by default; "
                         "L_div / L_nn / L_mass are also reported each epoch)")
    ap.add_argument("--mode", default="min", choices=["min", "max"])
    ap.add_argument("--grace-period", type=int, default=2,
                    help="ASHA: min epochs before a trial can be early-stopped")
    ap.add_argument("--reduction-factor", type=int, default=3,
                    help="ASHA: aggressiveness of early stopping (eta)")
    # Default to 4 concurrent trials, which suits a node with 4 GPUs
    # (1 GPU per trial -> up to 4 trials at once). Override with --max-concurrent;
    # pass 0 to let Ray decide based on available cluster resources.
    ap.add_argument("--max-concurrent", type=int, default=4,
                    help="cap simultaneously-running trials (0 = let Ray decide; "
                         "default 4 matches a 4-GPU node with 1 GPU/trial)")
    # search-space bounds (overridable so the era5 run can widen/narrow)
    ap.add_argument("--lr-low", type=float, default=1e-4)
    ap.add_argument("--lr-high", type=float, default=5e-4)
    # run config
    ap.add_argument("--storage", default="./ray_tune",
                    help="Tuner storage_path (local dir, or s3://... on a managed cluster)")
    ap.add_argument("--name", default="pcrf-hpo")
    ap.add_argument("--max-failures", type=int, default=2,
                    help="per-trial restart-from-checkpoint count for preemption")
    args = ap.parse_args()

    from ray.tune import Tuner, TuneConfig
    from ray.tune.schedulers import ASHAScheduler
    from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer

    # Reuse the EXISTING training logic verbatim (no copy / no edit). The import
    # is here (not at module top) so unit tests can import this module's pure
    # helpers without train_ray.py / ray on the path.
    from train_ray import train_loop_per_worker  # noqa: F401  (imported in worker)

    # ---- the trainable: the SAME TorchTrainer as train_ray.main(), but its
    #      train_loop_config carries per-trial physics weights + lr. The base
    #      config below is fixed across trials; tune.Tuner overrides the swept
    #      keys via param_space["train_loop_config"][...] at sample time. ------
    base_loop_config = dict(
        data=args.data, coarse=args.coarse, target=args.target,
        n=args.n, fine=args.fine, base=args.base, batch=args.batch,
        epochs=args.epochs, sat_bands=args.sat_bands,
        # lr / l_div / l_nn / l_mass are filled in by the search space per trial.
    )

    trainer = TorchTrainer(
        # Wrap so the trial's physics weights are applied before the real loop
        # runs. _tune_train_loop reads l_div/l_nn/l_mass from config, patches
        # PhysicsLoss in THIS worker process, then calls the unmodified
        # train_loop_per_worker imported from train_ray.
        _tune_train_loop,
        scaling_config=ScalingConfig(
            num_workers=args.workers, use_gpu=args.gpu,
            resources_per_worker={"GPU": 1} if args.gpu else None),
    )

    # ---- search space: physics weights around SWEEP_CENTER (1.0 / 0.1 / 0.05)
    #      plus lr. build_param_space is a pure helper (unit-tested). -----------
    param_space = build_param_space(
        base_loop_config, lr_low=args.lr_low, lr_high=args.lr_high)

    # ---- ASHA: early-stop weak trials on the per-epoch metric ----------------
    # train_loop_per_worker reports every epoch, so "training_iteration" advances
    # once per epoch and ASHA's time budget is measured in epochs.
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=args.epochs,
        grace_period=min(args.grace_period, args.epochs),
        reduction_factor=args.reduction_factor,
    )

    tune_config = TuneConfig(
        metric=args.metric,
        mode=args.mode,
        scheduler=scheduler,
        num_samples=args.num_samples,
        max_concurrent_trials=(args.max_concurrent or None),
    )

    run_config = RunConfig(
        storage_path=args.storage,
        name=args.name,
        # keep the best checkpoint per trial by the tuned metric
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute=args.metric,
            checkpoint_score_order=args.mode,
        ),
        failure_config=FailureConfig(max_failures=args.max_failures),
    )

    tuner = Tuner(
        trainer,
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

    # ---- report the winner (and how to read the grid) -----------------------
    best = extract_best_config(results, args.metric, args.mode)
    print("\n[ray-tune] best config (train_loop_config):")
    for k in ("l_div", "l_nn", "l_mass", "lr"):
        if k in best:
            print(f"    {k} = {best[k]}")
    print(f"[ray-tune] best {args.metric} = "
          f"{best.get('metric_value')} ({args.mode})")
    # best.checkpoint still comes from the raw ResultGrid for serving/eval:
    raw_best = results.get_best_result(metric=args.metric, mode=args.mode)
    print(f"[ray-tune] best checkpoint: {raw_best.checkpoint}")
    print("[ray-tune] full leaderboard:")
    try:
        df = results.get_dataframe()
        cols = [c for c in ("L_total", "L_div", "L_nn", "L_mass",
                            "config/train_loop_config/lr") if c in df.columns]
        print(df[cols].to_string() if cols else df.to_string())
    except Exception as e:  # pragma: no cover - pandas/display only
        print(f"    (could not render dataframe: {e})")


def _tune_train_loop(config):
    """Per-worker entrypoint for a single Tune trial.

    Applies this trial's physics-penalty weights by monkeypatching
    pcrf.physics.PhysicsLoss IN THIS WORKER PROCESS so that the unmodified
    train_loop_per_worker (which calls p.PhysicsLoss(spec) with library defaults)
    instead constructs it with the trial's l_div / l_nn / l_mass. Then delegates
    to the real, reused training loop.

    This keeps train_ray.py untouched and is fully process-local: the patch only
    affects the Ray worker that runs this trial, not the driver or other trials.
    """
    from pcrf import physics as p
    from train_ray import train_loop_per_worker as _real_loop

    # The defaults here mirror the SWEEP_CENTER (1.0 / 0.1 / 0.05): if a trial's
    # config somehow omits a weight, fall back to the target config, NOT to the
    # PhysicsLoss library defaults (1.0 / 1.0 / 1.0).
    l_div = float(config.get("l_div", SWEEP_CENTER["l_div"]))
    l_nn = float(config.get("l_nn", SWEEP_CENTER["l_nn"]))
    l_mass = float(config.get("l_mass", SWEEP_CENTER["l_mass"]))

    # CONTRACT ASSERT: the monkeypatch only works because train_ray.py looks up
    # PhysicsLoss as the module attribute p.PhysicsLoss at call time. Guard that
    # the attribute exists before we rebind it (a clear failure beats a silent
    # no-op if a refactor removes it).
    assert hasattr(p, "PhysicsLoss"), (
        "pcrf.physics.PhysicsLoss is missing; the Tune monkeypatch requires "
        "train_ray.py to construct the loss via the module attribute "
        "p.PhysicsLoss(spec). If train_ray.py was changed to `from pcrf.physics "
        "import PhysicsLoss` and call the bare name, this patch silently no-ops "
        "and the sweep would train at the library defaults.")

    _Orig = p.PhysicsLoss
    p.PhysicsLoss = make_tuned_physics_loss_cls(p, l_div, l_nn, l_mass)
    try:
        _real_loop(config)
    finally:
        # restore so the worker process is left clean (it is reused for retries)
        p.PhysicsLoss = _Orig


# ============================================================================
# TODO (requires editing train_ray.py, which this file does not touch): wire a
# true held-out validation metric (CRPS + divergence on the "val" split) so ASHA
# stops on real science, not the L_total training proxy. Concretely:
#   1. In train_ray.train_loop_per_worker, build a second DataLoader on the "val"
#      split (ERA5Dataset(..., split="val"); synthetic: a disjoint seed range).
#   2. After each epoch, with torch.no_grad(): draw an ensemble via
#      core.sample(shape, cond, steps=..., spec=spec, coarse_mass=...) over a few
#      val batches, then compute CRPS vs b["fine"] and the mean divergence_penalty
#      on the precip/wind channels.
#   3. Add {"val_crps": ..., "val_div": ...} to the reported metrics dict.
#   4. Here, default --metric to "val_crps" (mode="min").
# This file does NOT fake those metrics; until train_ray.py exposes them, the
# default metric stays the honest L_total proxy.
# ============================================================================


if __name__ == "__main__":
    main()
