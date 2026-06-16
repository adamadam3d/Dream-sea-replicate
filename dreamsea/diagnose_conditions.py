"""
Condition-signal diagnostic for DreamSea conditional generation.
==============================================================
CPU-only. Answers one question: do the 2D PCA conditions carry enough signal
for the conditional model to learn a "type" from them, or are they near-
meaningless (which would explain a model that ignores the condition and always
produces its average output)?

It reports two things:
  1. PCA explained-variance ratio — how much of the DINOv2 feature variance the
     2 kept components actually capture. Low here => the condition is a near-
     constant number across images => nothing to condition on (DATA problem).
  2. Spread of the saved per-image conditions — confirms the conditions are (or
     aren't) numerically distinct, and how far they sit from the [0,0] origin
     the model is strongest at.

Usage:
    # Point at a preprocess_dataset.py output dir (has pca_model.pkl + conditions/)
    python -m dreamsea.diagnose_conditions --data_dir path/to/preprocess_out

    # Or specify the two paths explicitly
    python -m dreamsea.diagnose_conditions \
        --pca_model_path .../pca_model.pkl --conditions_dir .../conditions
"""

import argparse
from pathlib import Path

import numpy as np


def _load_pca(pca_model_path):
    import joblib
    pca = joblib.load(pca_model_path)
    return pca


def _load_conditions(conditions_dir):
    """Stack every conditions/<name>_cond.pt into an (N, 2) array."""
    import torch
    cond_dir = Path(conditions_dir)
    pt_files = sorted(cond_dir.glob("*_cond.pt"))
    if not pt_files:
        pt_files = sorted(cond_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No condition .pt files found in {cond_dir}")

    vecs = []
    for f in pt_files:
        t = torch.load(f, map_location="cpu", weights_only=True)
        vecs.append(np.asarray(t.float().numpy(), dtype=np.float64).reshape(-1))
    return np.stack(vecs, axis=0), [f.name for f in pt_files]


def _report_pca(pca):
    print("\n=== 1. PCA explained variance ===")
    evr = getattr(pca, "explained_variance_ratio_", None)
    if evr is None:
        print("  PCA object has no explained_variance_ratio_ (not fitted?).")
        return None
    n = getattr(pca, "n_components_", len(evr))
    print(f"  Kept components: {n}")
    for i, r in enumerate(evr):
        print(f"    PC{i + 1}: {r * 100:6.2f}% of variance")
    cum = float(np.sum(evr))
    print(f"  Total captured by the {len(evr)} kept components: {cum * 100:.2f}%")

    print("\n  Interpretation:")
    if cum < 0.15:
        print(f"  ⚠ Only {cum*100:.1f}% captured. The 2 condition numbers barely distinguish")
        print("    images — likely a homogeneous dataset. The model has almost no 'type'")
        print("    signal to learn from. This alone explains no type match (DATA problem).")
    elif cum < 0.40:
        print(f"  ~ {cum*100:.1f}% captured. Weak-to-moderate signal. The condition carries")
        print("    something, but a thin cross_attention_dim=2 pathway + dropout may be")
        print("    losing it. Suspect BOTH data and architecture.")
    else:
        print(f"  ✓ {cum*100:.1f}% captured. The condition is reasonably informative, so")
        print("    'no type match' points at the MODEL (thin conditioning / dropout /")
        print("    undertraining) rather than the data.")
    return cum


def _report_conditions(conds, names):
    print("\n=== 2. Saved condition spread ===")
    N = conds.shape[0]
    print(f"  Samples: {N}")
    mean = conds.mean(axis=0)
    std = conds.std(axis=0)
    cmin = conds.min(axis=0)
    cmax = conds.max(axis=0)
    print(f"  Per-axis mean: [{mean[0]:+.4f}, {mean[1]:+.4f}]  (expected ~0 for PCA)")
    print(f"  Per-axis std:  [{std[0]:.4f}, {std[1]:.4f}]")
    print(f"  Per-axis min:  [{cmin[0]:+.4f}, {cmin[1]:+.4f}]")
    print(f"  Per-axis max:  [{cmax[0]:+.4f}, {cmax[1]:+.4f}]")

    # How clustered are conditions? Compare typical pairwise distance to the
    # per-sample distance from the origin. If conditions are tightly clustered,
    # the model sees nearly the same input for every image.
    dists = np.linalg.norm(conds - mean, axis=1)
    print(f"\n  Distance from centroid: mean={dists.mean():.4f}, "
          f"median={np.median(dists):.4f}, max={dists.max():.4f}")

    # Fraction sitting very close to the [0,0] origin (where the model is
    # strongest); if most conditions cluster there, off-origin generation is
    # both rare in training and the weakest regime at inference.
    origin_dist = np.linalg.norm(conds, axis=1)
    typical = np.median(origin_dist) + 1e-9
    near_frac = float(np.mean(origin_dist < 0.25 * typical))
    print(f"  Distance from [0,0] origin: median={np.median(origin_dist):.4f}, "
          f"max={origin_dist.max():.4f}")

    # Closest pair — duplicate-ish conditions mean different images are asked of
    # the model under identical conditioning.
    if N <= 4000:
        diffs = conds[:, None, :] - conds[None, :, :]
        d = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        print(f"  Closest pair: {names[i]} <-> {names[j]}  (distance {d[i, j]:.4f})")

    return std


def main():
    parser = argparse.ArgumentParser(description="Diagnose DreamSea condition signal (CPU-only).")
    parser.add_argument("-i", "--data_dir", type=str, default=None,
                        help="preprocess_dataset.py output dir (contains pca_model.pkl and conditions/).")
    parser.add_argument("-p", "--pca_model_path", type=str, default=None,
                        help="Explicit path to pca_model.pkl (overrides --data_dir).")
    parser.add_argument("-c", "--conditions_dir", type=str, default=None,
                        help="Explicit path to the conditions/ folder (overrides --data_dir).")
    args = parser.parse_args()

    pca_path = args.pca_model_path
    cond_dir = args.conditions_dir
    if args.data_dir:
        base = Path(args.data_dir)
        pca_path = pca_path or str(base / "pca_model.pkl")
        cond_dir = cond_dir or str(base / "conditions")

    if not pca_path and not cond_dir:
        parser.error("Provide --data_dir, or --pca_model_path and/or --conditions_dir.")

    cum = None
    if pca_path and Path(pca_path).exists():
        cum = _report_pca(_load_pca(pca_path))
    elif pca_path:
        print(f"\n[skip] PCA model not found at {pca_path}")

    cond_std = None
    if cond_dir and Path(cond_dir).exists():
        conds, names = _load_conditions(cond_dir)
        cond_std = _report_conditions(conds, names)
    elif cond_dir:
        print(f"\n[skip] conditions dir not found at {cond_dir}")

    print("\n=== Verdict ===")
    if cum is not None and cum < 0.15:
        print("  Root cause is most likely the DATA: the 2D condition is near-constant")
        print("  across images, so the model cannot learn a type. Fixes (all need a")
        print("  retrain): more PCA components, spatial DINO features instead of CLS,")
        print("  or more diverse data.")
    elif cum is not None and cum >= 0.40:
        print("  The condition is informative, so 'no type match' points at the MODEL.")
        print("  Fixes (retrain): embed the 2D condition via an MLP to a larger")
        print("  cross_attention_dim, add cross-attn at more resolutions, set")
        print("  cond_dropout_prob=0.")
    else:
        print("  Mixed/weak signal — suspect both. Start by raising PCA components and")
        print("  thickening the conditioning pathway, then retrain.")


if __name__ == "__main__":
    main()
