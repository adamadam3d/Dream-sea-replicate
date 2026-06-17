"""
Latent-control accuracy evaluation (paper Table 1 / Figure 13).
==============================================================
Measures whether the conditional model actually generates "the type you asked
for", using the paper's round-trip metric in DINOv2/PCA embedding space:

    target latent  phi
        -> generate_patch(phi)              # model draws an RGBD image
        -> RGB -> DINOv2 -> PCA -> phi'      # re-encode it the SAME way training did
        -> error = (phi - phi')

If conditioning works, an image generated from phi reads back as ~phi, so the
error is small. If the model ignores the condition, phi' is ~constant (the
dataset mean) regardless of phi, so the error is large.

Two modes:
  * single checkpoint  (-c ckpt.pt)         -> metrics + patch collage + scatter
  * whole folder       (-C checkpoints_dir) -> evaluate EVERY matching checkpoint
    against the same targets and emit a comparison table + a trend plot, so you
    can see latent control improve over training epochs / pick the best ckpt.

Stricter than measure_conditioning.py: that checks different conditions give
different outputs (variance); this checks the mapping is CORRECT (you get back
what you asked for).

Requires a GPU (diffusion sampling) and DINOv2 weights.

Usage:
    # single checkpoint
    python -m dreamsea.eval_latent_control -c checkpoints/cond.pt \
        --conditions_dir <preprocess_out>/conditions \
        --pca_model_path <preprocess_out>/pca_model.pkl -k 16

    # compare every conditional checkpoint in a folder
    python -m dreamsea.eval_latent_control -C checkpoints \
        --conditions_dir <preprocess_out>/conditions \
        --pca_model_path <preprocess_out>/pca_model.pkl -k 16
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dreamsea.generation_inpainting import GeneratorInpainter


def _load_dino(device):
    """Load DINOv2-large exactly as data_preprocessing.py does, for re-encoding.

    Kept self-contained (not via DataPreprocessor) so this eval doesn't also pull
    in the Depth-Anything model it doesn't need.
    """
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def _encode_latent(rgb_chw, processor, model, pca, device):
    """RGB patch in [-1,1] (3,H,W) -> 2D PCA latent, mirroring training's
    extract_and_reduce_dino_features (CLS token -> fitted PCA)."""
    rgb = np.clip((rgb_chw[:3] + 1.0) / 2.0, 0.0, 1.0)
    img = Image.fromarray((np.transpose(rgb, (1, 2, 0)) * 255.0).astype(np.uint8), mode="RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    cls = model(**inputs).last_hidden_state[0, 0, :].cpu().numpy()
    return pca.transform(cls[None, :])[0].astype(np.float32)


def _pick_targets(conditions_dir, k, rng):
    """Sample k real conditions uniformly from the preprocessed set."""
    pt_files = sorted(Path(conditions_dir).glob("*_cond.pt"))
    if not pt_files:
        pt_files = sorted(Path(conditions_dir).glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No condition .pt files in {conditions_dir}")
    idx = rng.choice(len(pt_files), size=min(k, len(pt_files)), replace=False)
    targets, names = [], []
    for i in idx:
        t = torch.load(pt_files[i], map_location="cpu", weights_only=True)
        targets.append(np.asarray(t.float().numpy(), dtype=np.float32).reshape(-1))
        names.append(pt_files[i].stem.replace("_cond", ""))
    return np.stack(targets, axis=0), names


def _pearson(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _parse_epoch(path):
    """Best-effort epoch number from a checkpoint filename (last integer)."""
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else -1


def _evaluate_checkpoint(ckpt_path, targets, names, std_mu, std_sd, pca, processor, dino, args):
    """Generate from each target latent, re-encode, and score one checkpoint.

    std_mu/std_sd are fixed (computed once from the target set) so standardized
    MSE is comparable across checkpoints. Seeds are reset here so every
    checkpoint sees the same initial noise — differences come from the weights.
    """
    inpainter = GeneratorInpainter(cond_model_path=str(ckpt_path), uncond_model_path=None, device=args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    K = targets.shape[0]
    preds = np.zeros_like(targets)
    patches = []
    for i in range(K):
        patch = inpainter.generate_patch(targets[i], num_inference_steps=args.num_inference_steps)[0]
        preds[i] = _encode_latent(patch, processor, dino, pca, args.device)
        patches.append(patch)

    raw_mse_axis = ((targets - preds) ** 2).mean(axis=0)
    ts = (targets - std_mu) / std_sd
    ps = (preds - std_mu) / std_sd
    std_mse_axis = ((ts - ps) ** 2).mean(axis=0)
    r_axis = [_pearson(targets[:, j], preds[:, j]) for j in range(targets.shape[1])]

    metrics = {
        "checkpoint": str(ckpt_path),
        "epoch": _parse_epoch(ckpt_path),
        "raw_mse": float(raw_mse_axis.mean()),
        "raw_mse_per_axis": raw_mse_axis.tolist(),
        "standardized_mse": float(std_mse_axis.mean()),
        "standardized_mse_per_axis": std_mse_axis.tolist(),
        "correlation_per_axis": r_axis,
        "mean_abs_correlation": float(np.mean(np.abs(r_axis))),
        "targets": targets.tolist(),
        "measured": preds.tolist(),
        "names": names,
    }
    return metrics, patches


def _interpret(std_mse):
    if std_mse < 0.5:
        return "strong control (<0.5)"
    if std_mse < 0.9:
        return "partial control (0.5-0.9)"
    return "little/no control (>=0.9)"


def _save_single_visuals(metrics, patches, out_dir, tag):
    """Patch collage + target-vs-measured scatter for a single checkpoint."""
    targets = np.array(metrics["targets"], dtype=np.float32)
    preds = np.array(metrics["measured"], dtype=np.float32)
    r_axis = metrics["correlation_per_axis"]

    cell = 224
    collage = Image.new("RGB", (len(patches) * cell, cell))
    for i, patch in enumerate(patches):
        rgb = np.clip((patch[:3] + 1.0) / 2.0, 0.0, 1.0)
        collage.paste(Image.fromarray((np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)), (i * cell, 0))
    collage_path = os.path.join(out_dir, f"latent_control_{tag}.png")
    collage.save(collage_path)
    print(f"Generated-patch collage: {collage_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = targets.shape[1]
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        for j, ax in enumerate(np.atleast_1d(axes)):
            ax.scatter(targets[:, j], preds[:, j], alpha=0.7)
            lo = min(targets[:, j].min(), preds[:, j].min())
            hi = max(targets[:, j].max(), preds[:, j].max())
            ax.plot([lo, hi], [lo, hi], "r--", label="perfect")
            ax.set_xlabel(f"target PC{j+1}"); ax.set_ylabel(f"measured PC{j+1}")
            ax.set_title(f"PC{j+1}  (r={r_axis[j]:.2f})"); ax.legend()
        fig.tight_layout()
        scatter_path = os.path.join(out_dir, f"latent_control_{tag}_scatter.png")
        fig.savefig(scatter_path, dpi=120)
        print(f"Target-vs-measured scatter: {scatter_path}")
    except Exception as e:
        print(f"(skipped scatter plot: {e})")


def _save_comparison(rows, out_dir):
    """CSV + JSON + trend plot comparing all checkpoints (sorted by epoch)."""
    rows = sorted(rows, key=lambda r: r["epoch"])
    csv_path = os.path.join(out_dir, "latent_control_comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "epoch", "standardized_mse", "raw_mse", "mean_abs_correlation"])
        for r in rows:
            w.writerow([Path(r["checkpoint"]).name, r["epoch"],
                        f"{r['standardized_mse']:.4f}", f"{r['raw_mse']:.4f}",
                        f"{r['mean_abs_correlation']:.4f}"])
    print(f"\nComparison CSV: {csv_path}")

    json_path = os.path.join(out_dir, "latent_control_comparison.json")
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"Comparison JSON: {json_path}")

    print("\n=== Checkpoint comparison (sorted by epoch) ===")
    print(f"  {'checkpoint':<34} {'epoch':>6} {'std_MSE':>9} {'mean|r|':>9}  verdict")
    best = min(rows, key=lambda r: r["standardized_mse"])
    for r in rows:
        star = "  <-- best" if r is best else ""
        print(f"  {Path(r['checkpoint']).name:<34} {r['epoch']:>6} "
              f"{r['standardized_mse']:>9.4f} {r['mean_abs_correlation']:>9.4f}  {_interpret(r['standardized_mse'])}{star}")
    print(f"\n  Best latent control: {Path(best['checkpoint']).name} "
          f"(std_MSE={best['standardized_mse']:.4f}, mean|r|={best['mean_abs_correlation']:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [r["epoch"] for r in rows]
        smse = [r["standardized_mse"] for r in rows]
        mr = [r["mean_abs_correlation"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(ep, smse, "o-", color="tab:red", label="standardized MSE (lower=better)")
        ax1.axhline(1.0, ls=":", color="gray", lw=1, label="ignores condition (MSE=1)")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("standardized MSE", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(ep, mr, "s--", color="tab:blue", label="mean |correlation| (higher=better)")
        ax2.set_ylabel("mean |correlation|", color="tab:blue"); ax2.set_ylim(0, 1)
        ax1.set_title("Latent-control accuracy over checkpoints")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
        fig.tight_layout()
        plot_path = os.path.join(out_dir, "latent_control_trend.png")
        fig.savefig(plot_path, dpi=120)
        print(f"Trend plot: {plot_path}")
    except Exception as e:
        print(f"(skipped trend plot: {e})")


def main():
    parser = argparse.ArgumentParser(description="Latent-control accuracy (paper Table 1 round-trip MSE).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-c", "--cond_model_path", type=str, help="Single conditional checkpoint to evaluate.")
    src.add_argument("-C", "--checkpoints_dir", type=str, help="Folder of checkpoints; evaluate and compare all matching ones.")
    parser.add_argument("--pattern", type=str, default="conditional*.pt",
                        help="Glob for checkpoints inside --checkpoints_dir (default: conditional*.pt).")
    parser.add_argument("-i", "--conditions_dir", type=str, default=None, help="Preprocessed conditions/ dir to sample targets from.")
    parser.add_argument("-p", "--pca_model_path", type=str, default=None,
                        help="Fitted pca_model.pkl to re-encode generated images. If omitted, looked for next to conditions_dir's parent.")
    parser.add_argument("-k", "--num_targets", type=int, default=16, help="Number of target latents to evaluate.")
    parser.add_argument("-n", "--num_inference_steps", type=int, default=200, help="Denoising steps per sample.")
    parser.add_argument("-o", "--output_dir", type=str, default="outputs/latent_control", help="Where to save reports/plots.")
    parser.add_argument("-x", "--seed", type=int, default=0, help="RNG seed (target sampling + generation).")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Resolve PCA model: explicit, else next to the conditions dir's parent.
    pca_path = args.pca_model_path
    if not pca_path and args.conditions_dir:
        cand = Path(args.conditions_dir).parent / "pca_model.pkl"
        if cand.exists():
            pca_path = str(cand)
    if not pca_path or not os.path.exists(pca_path):
        parser.error("Need --pca_model_path (the fitted PCA from preprocessing) to re-encode generated images.")
    if not args.conditions_dir:
        parser.error("Need --conditions_dir to sample target latents.")

    import joblib
    pca = joblib.load(pca_path)
    targets, names = _pick_targets(args.conditions_dir, args.num_targets, rng)
    K = targets.shape[0]
    # Fixed standardization stats so scores are comparable across checkpoints.
    std_mu = targets.mean(axis=0)
    std_sd = targets.std(axis=0) + 1e-8
    print(f"Evaluating {K} target latents from {args.conditions_dir}")

    print("Loading DINOv2 for re-encoding...")
    processor, dino = _load_dino(args.device)

    # ---- Folder mode: evaluate & compare every matching checkpoint ----
    if args.checkpoints_dir:
        ckpts = sorted(Path(args.checkpoints_dir).glob(args.pattern), key=_parse_epoch)
        if not ckpts:
            parser.error(f"No checkpoints matching '{args.pattern}' in {args.checkpoints_dir}")
        print(f"Found {len(ckpts)} checkpoints matching '{args.pattern}'.")
        rows = []
        for ci, ckpt in enumerate(ckpts):
            print(f"\n[{ci+1}/{len(ckpts)}] {ckpt.name}")
            try:
                metrics, _ = _evaluate_checkpoint(ckpt, targets, names, std_mu, std_sd, pca, processor, dino, args)
            except Exception as e:
                print(f"  [skip] failed to evaluate {ckpt.name}: {e}")
                continue
            print(f"  std_MSE={metrics['standardized_mse']:.4f}  "
                  f"mean|r|={metrics['mean_abs_correlation']:.3f}  ({_interpret(metrics['standardized_mse'])})")
            rows.append(metrics)
        if not rows:
            print("No checkpoints evaluated successfully.")
            return
        _save_comparison(rows, args.output_dir)
        return

    # ---- Single-checkpoint mode ----
    print(f"Loading checkpoint: {args.cond_model_path}")
    metrics, patches = _evaluate_checkpoint(args.cond_model_path, targets, names, std_mu, std_sd, pca, processor, dino, args)
    for i in range(K):
        print(f"  [{i+1}/{K}] {names[i]}: target={np.round(targets[i],2).tolist()} "
              f"-> measured={np.round(metrics['measured'][i],2).tolist()}")
    print("\n=== Latent-control accuracy ===")
    print(f"  Raw PCA-space MSE   : {metrics['raw_mse']:.4f}   (per-axis {np.round(metrics['raw_mse_per_axis'],4).tolist()})")
    print(f"  Standardized MSE    : {metrics['standardized_mse']:.4f}   (per-axis {np.round(metrics['standardized_mse_per_axis'],3).tolist()})")
    print(f"  Target<->measured r : {[round(r,3) for r in metrics['correlation_per_axis']]}")
    print(f"\n  Verdict (0=perfect, 1=ignores condition, 2=independent): {_interpret(metrics['standardized_mse'])}")
    print("  Compare OLD vs NEW checkpoints — lower MSE / higher r = improvement.")

    tag = Path(args.cond_model_path).stem
    _save_single_visuals(metrics, patches, args.output_dir, tag)
    report_path = os.path.join(args.output_dir, f"latent_control_{tag}.json")
    with open(report_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
