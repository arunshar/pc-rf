"""
==============================================================================
 PC-RF physics-penalty sanity check  (SYNTHETIC DATA -- not ERA5/Sentinel-2)
==============================================================================

PURPOSE
  Check that the three physics-penalty operators written in this script
  actually reduce the physics-violation metrics they target, and that doing so
  does NOT destroy the underlying signal. This is a *mechanism* check on toy
  fields, run with pure numpy. It produces REAL numbers from REAL computation.

  These numbers are SYNTHETIC. They say nothing about ERA5 or Sentinel-2 data.
  This script does not import pcrf, so it does not test src/pcrf/physics.py.
  Use the PyTorch pipeline in src/pcrf on real data for real results.

WHAT IT DOES
  1. Builds a physically valid "ground truth":
       - wind (u, v) that is divergence-free by construction (from a stream
         function psi: u = d psi/dy, v = -d psi/dx)
       - precipitation p that is non-negative (sum of Gaussian blobs)
       - a known coarse-resolution mass (block-mean of p)
  2. Builds an "RF-base" prediction = truth + correlated noise + negative bias,
     mimicking what a generator WITHOUT physics constraints produces.
  3. Builds a "PC-RF" prediction by minimizing the combined physics objective
     (L_div + L_nn + L_mass) plus a data-fidelity term, via gradient descent
     with analytic gradients (periodic central differences -> exact adjoints).
  4. Reports DE / NVR / MCE (physics) and RMSE / correlation / SSIM-proxy
     (fidelity) for both, and saves a figure.

Run:  python3 synthetic_sanity_check.py
Out:  results/synthetic_metrics.md
      results/synthetic_physics_demo.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RNG = np.random.default_rng(0)          # fixed seed -> reproducible
H = W = 128                             # fine grid
BLOCK = 8                               # coarse = HxW / BLOCK
os.makedirs("results", exist_ok=True)


# ----------------------------------------------------------------------------
# Periodic central-difference operators (skew-adjoint: D^T = -D)
# ----------------------------------------------------------------------------
def dx(f):  return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / 2.0
def dy(f):  return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / 2.0
def dxT(g): return -dx(g)
def dyT(g): return -dy(g)

def divergence(u, v):
    return dx(u) + dy(v)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def divergence_error(u, v):
    """Mean absolute divergence (DE). Lower = more physically valid wind."""
    return float(np.mean(np.abs(divergence(u, v))))

def nonneg_violation_rate(p):
    """Fraction of precip cells that are negative (NVR)."""
    return float(np.mean(p < 0.0))

def mass_conservation_error(p, coarse_mean):
    """Relative error between fine-field mean and coarse mass (MCE)."""
    return float(abs(p.mean() - coarse_mean) / (abs(coarse_mean) + 1e-12))

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def corr(a, b):
    af, bf = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))

def box_filter(img, k):
    """Exact k x k mean filter via integral image (periodic-safe enough)."""
    pad = k // 2
    P = np.pad(img, pad, mode="reflect")
    C = np.cumsum(np.cumsum(P, axis=0), axis=1)
    C = np.pad(C, ((1, 0), (1, 0)), mode="constant")
    out = (C[k:, k:] - C[:-k, k:] - C[k:, :-k] + C[:-k, :-k]) / (k * k)
    return out[:img.shape[0], :img.shape[1]]

def ssim_proxy(a, b, k=7, L=None):
    """Simple windowed SSIM (numpy). Fidelity proxy, not the paper metric."""
    if L is None:
        L = max(a.max() - a.min(), 1e-6)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_a, mu_b = box_filter(a, k), box_filter(b, k)
    va = box_filter(a * a, k) - mu_a ** 2
    vb = box_filter(b * b, k) - mu_b ** 2
    vab = box_filter(a * b, k) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2) + 1e-12)
    return float(np.clip(s, -1, 1).mean())


# ----------------------------------------------------------------------------
# Synthetic ground truth
# ----------------------------------------------------------------------------
def smooth_field(scale=6.0):
    """Low-pass random field via FFT spectral filtering."""
    noise = RNG.standard_normal((H, W))
    fx = np.fft.fftfreq(W)[None, :]
    fy = np.fft.fftfreq(H)[:, None]
    f2 = fx ** 2 + fy ** 2
    flt = np.exp(-f2 * (scale ** 2) * 20.0)
    out = np.real(np.fft.ifft2(np.fft.fft2(noise) * flt))
    return (out - out.mean()) / (out.std() + 1e-9)

def make_truth():
    # Divergence-free wind from a stream function psi:  u =  d psi/dy,  v = -d psi/dx
    psi = smooth_field(scale=7.0)
    u = dy(psi)
    v = -dx(psi)
    # Non-negative precipitation = sum of Gaussian blobs
    yy, xx = np.mgrid[0:H, 0:W]
    p = np.zeros((H, W))
    for _ in range(9):
        cy, cx = RNG.integers(0, H), RNG.integers(0, W)
        r = RNG.uniform(6, 16)
        amp = RNG.uniform(0.5, 1.5)
        p += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    return u, v, p

def coarse_mean_of(p):
    """Block-mean down to coarse grid, then the scalar domain mass."""
    c = p.reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK).mean(axis=(1, 3))
    return float(c.mean())


# ----------------------------------------------------------------------------
# RF-base proxy (unconstrained generator output)
# ----------------------------------------------------------------------------
def make_rfbase(u, v, p):
    nu = u + 0.35 * smooth_field(4.0) * u.std()
    nv = v + 0.35 * smooth_field(4.0) * v.std()
    npp = p + 0.30 * smooth_field(4.0) * p.std() - 0.18 * p.std()  # bias -> negatives
    return nu, nv, npp


# ----------------------------------------------------------------------------
# PC-RF refinement: minimize  L = wd*data + l1*L_div + l2*L_nn + l3*L_mass
# (gradient descent with analytic gradients), then an inference-time physics
# projection (strict non-negativity + exact mass conservation), exactly as the
# paper describes (softplus / non-neg projection on the precip channel).
#
# Step size note: the divergence Hessian 2 D^T D has max eigenvalue ~4 per mode
# (s_x=s_y=1), so stability needs lr*(wd + 4*l1) < 2. We use lr=0.05, l1=2.0
# -> lr*(1 + 8) = 0.45 < 1, comfortably stable.
# ----------------------------------------------------------------------------
def pcrf_refine(u0, v0, p0, coarse_mass,
                steps=600, lr=0.05, wd=1.0, l1=2.0, l2=4.0, l3=6.0,
                project=True):
    u, v, p = u0.copy(), v0.copy(), p0.copy()
    N = p.size
    hist = []
    for t in range(steps):
        # --- divergence penalty: grad_u sum(div^2) = 2 dx^T(div) ---
        d = divergence(u, v)
        gu_div = 2 * dxT(d)
        gv_div = 2 * dyT(d)
        # --- non-negativity penalty (soft) ---
        gp_nn = 2 * np.minimum(p, 0.0)
        # --- mass conservation penalty ---
        mass_gap = p.mean() - coarse_mass
        gp_mass = 2 * mass_gap * np.ones_like(p) / N
        # --- data fidelity (stay near the generator output) ---
        gu_dat, gv_dat, gp_dat = (u - u0), (v - v0), (p - p0)
        # --- gradient step ---
        u -= lr * (wd * gu_dat + l1 * gu_div)
        v -= lr * (wd * gv_dat + l1 * gv_div)
        p -= lr * (wd * gp_dat + l2 * gp_nn + l3 * gp_mass)
        if t % 20 == 0:
            hist.append((t, divergence_error(u, v),
                         nonneg_violation_rate(p),
                         mass_conservation_error(p, coarse_mass)))
    # --- inference-time physics projection on the precip channel ---
    if project:
        p = np.maximum(p, 0.0)                       # strict non-negativity
        p *= coarse_mass / (p.mean() + 1e-12)        # exact mass (keeps p>=0)
    return u, v, p, hist


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def metrics_row(name, u, v, p, truth_p, cmass):
    return {
        "method": name,
        "DE":   divergence_error(u, v),
        "NVR":  nonneg_violation_rate(p),
        "MCE":  mass_conservation_error(p, cmass),
        "RMSE": rmse(p, truth_p),
        "Corr": corr(p, truth_p),
        "SSIM": ssim_proxy(p, truth_p),
    }


def main():
    u_t, v_t, p_t = make_truth()
    cmass = coarse_mean_of(p_t)

    u_b, v_b, p_b = make_rfbase(u_t, v_t, p_t)
    u_p, v_p, p_p, hist = pcrf_refine(u_b, v_b, p_b, cmass)

    rows = [
        metrics_row("Ground truth", u_t, v_t, p_t, p_t, cmass),
        metrics_row("RF-base (no physics)", u_b, v_b, p_b, p_t, cmass),
        metrics_row("PC-RF (physics)", u_p, v_p, p_p, p_t, cmass),
    ]

    # ---- print + save markdown table ----
    hdr = f"{'method':22s} {'DE↓':>9s} {'NVR↓':>8s} {'MCE↓':>9s} " \
          f"{'RMSE↓':>8s} {'Corr↑':>7s} {'SSIM↑':>7s}"
    print("\n" + "=" * 74)
    print("  SYNTHETIC physics-penalty sanity check  (NOT ERA5 results)")
    print("=" * 74)
    print(hdr)
    print("-" * 74)
    lines = ["# Synthetic physics-penalty sanity check (NOT ERA5 results)\n",
             "Pure-numpy mechanism check on toy fields. The PC-RF row is one fixed field",
             "refined by gradient descent on the penalties and then projected, so its",
             "gain cannot be split between the penalties and the projection.",
             "These numbers are SYNTHETIC. They say nothing about ERA5 or trained models.\n",
             "| Method | DE down | NVR down | MCE down | RMSE down | Corr up | SSIM up |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        print(f"{r['method']:22s} {r['DE']:9.4f} {r['NVR']:8.3f} {r['MCE']:9.4f} "
              f"{r['RMSE']:8.4f} {r['Corr']:7.3f} {r['SSIM']:7.3f}")
        lines.append(f"| {r['method']} | {r['DE']:.4f} | {r['NVR']:.3f} | "
                     f"{r['MCE']:.4f} | {r['RMSE']:.4f} | {r['Corr']:.3f} | "
                     f"{r['SSIM']:.3f} |")
    print("=" * 74)

    # ---- improvement summary ----
    base, pcrf = rows[1], rows[2]
    def pct(a, b):  # reduction %
        return 100.0 * (a - b) / (abs(a) + 1e-12)
    print("\nPC-RF vs RF-base (synthetic):")
    print(f"  divergence error  : {pct(base['DE'], pcrf['DE']):5.1f}% lower")
    print(f"  non-neg violations: {base['NVR']:.3f} -> {pcrf['NVR']:.3f}")
    print(f"  mass cons. error  : {pct(base['MCE'], pcrf['MCE']):5.1f}% lower")
    print(f"  correlation       : {base['Corr']:.3f} -> {pcrf['Corr']:.3f}\n")
    lines += ["",
              f"**PC-RF vs RF-base (synthetic):** divergence error "
              f"{pct(base['DE'], pcrf['DE']):.1f}% lower, "
              f"NVR {base['NVR']:.3f} to {pcrf['NVR']:.3f}, "
              f"MCE {pct(base['MCE'], pcrf['MCE']):.1f}% lower, "
              f"correlation {base['Corr']:.3f} to {pcrf['Corr']:.3f}."]

    with open("results/synthetic_metrics.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure ----
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2))
    vmax_p = p_t.max()

    def show(ax, img, title, cmap="viridis", v=None):
        im = ax.imshow(img, cmap=cmap,
                       vmin=(0 if v else None), vmax=(v if v else None))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axs[0, 0], p_t, "Truth precip (non-neg)", v=vmax_p)
    show(axs[0, 1], p_b, "RF-base precip", v=vmax_p)
    show(axs[0, 2], p_p, "PC-RF precip", v=vmax_p)
    axs[0, 3].imshow((p_b < 0), cmap="Reds")
    axs[0, 3].set_title(f"RF-base negative cells\nNVR={nonneg_violation_rate(p_b):.3f}",
                        fontsize=10)
    axs[0, 3].axis("off")

    show(axs[1, 0], np.abs(divergence(u_t, v_t)), "Truth |div|", cmap="magma")
    show(axs[1, 1], np.abs(divergence(u_b, v_b)), "RF-base |div|", cmap="magma")
    show(axs[1, 2], np.abs(divergence(u_p, v_p)), "PC-RF |div|", cmap="magma")

    hist = np.array(hist)
    ax = axs[1, 3]
    ax.plot(hist[:, 0], hist[:, 1] / hist[0, 1], color="#1d6fb8", lw=2,
            label="DE (soft penalty)")
    ax.set_title("Divergence error during\nsoft-penalty refinement", fontsize=10)
    ax.set_xlabel("iteration"); ax.set_ylabel("DE relative to start")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.40,
            "NVR and MCE are enforced\nexactly by the inference\n"
            "projection (NVR: 0.236$\\to$0,\nMCE$\\to$0). See table.",
            transform=ax.transAxes, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff4ec", ec="#e76f51"))
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("PC-RF physics-penalty sanity check  (SYNTHETIC toy fields, "
                 "pure numpy -- NOT ERA5/Sentinel-2 results)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("results/synthetic_physics_demo.png", dpi=150)
    print("wrote results/synthetic_metrics.md and "
          "results/synthetic_physics_demo.png\n")


if __name__ == "__main__":
    main()
