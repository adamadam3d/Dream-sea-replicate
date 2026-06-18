"""
DreamSea RGBD Viewer
====================
Interactive viewer to compare RGB images with their estimated depth maps.

Usage:
    # View a single .pt tensor:
    python -m dreamsea.view_rgbd --file path/to/sample_rgbd.pt

    # View preprocessed .pt tensors:
    python -m dreamsea.view_rgbd --data_dir path/to/preprocessed_output

    # View raw images (runs depth estimation live):
    python -m dreamsea.view_rgbd --raw_dir path/to/raw_images --device cuda

Running over SSH / headless (no display):
    The viewer auto-detects a missing display and renders to PNG instead of
    opening a GUI window. Force it explicitly with --save:
        python -m dreamsea.view_rgbd --file sample_rgbd.pt --save sample.png
    Then open the PNG in VS Code Remote, or scp it to your local machine.
    (For an interactive window over SSH instead, use `ssh -X` so DISPLAY is set.)

Controls (interactive mode only):
    Left/Right, A/D, or Scroll — Navigate between images
    Home/End                   — Jump to first/last image
    C                          — Toggle depth colormap (magma/viridis/inferno/plasma)
    S                          — Save current side-by-side figure as PNG
    Q or Escape                — Quit
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
# Default to the headless-safe 'Agg' backend so importing never fails on a
# display-less SSH session. __main__ switches to an interactive backend only
# when a display is actually available (see below).
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch


# ─── Color map rotation ──────────────────────────────────────────────
DEPTH_CMAPS = ['magma', 'viridis', 'inferno', 'plasma', 'cividis', 'turbo']


class RGBDViewer:
    """Interactive side-by-side RGB vs Depth viewer with lazy loading."""

    def __init__(self, paths: list, names: list, output_dir: str = None, save_path: str = None):
        """
        Args:
            paths: List of file paths to [4, H, W] .pt tensors (loaded on demand).
            names: List of display names for each tensor.
            output_dir: Optional directory to save interactive screenshots into.
            save_path: If set, run headless — render each tensor to a PNG here
                (a .png path for a single tensor, otherwise a directory that
                receives one <name>_view.png per tensor) and skip the GUI.
        """
        self.paths = paths
        self.names = names
        self.count = len(paths)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.save_path = save_path
        self.idx = 0
        self.cmap_idx = 0

        # ── Set up figure ──
        self.fig = plt.figure(figsize=(14, 6), facecolor='#1a1a2e')
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title('DreamSea RGBD Viewer')

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)
        self.ax_rgb = self.fig.add_subplot(gs[0])
        self.ax_depth = self.fig.add_subplot(gs[1])

        # Title
        self.suptitle = self.fig.suptitle(
            '', fontsize=13, fontweight='bold', color='#e0e0e0', y=0.97
        )

        # Footer help text
        self.fig.text(
            0.5, 0.01,
            '← → or Scroll to Navigate  |  C Colormap  |  S Save  |  Q Quit',
            ha='center', va='bottom', fontsize=9, color='#888888',
            fontstyle='italic'
        )

        # Placeholders for images
        self.im_rgb = None
        self.im_depth = None
        self.cbar = None

        # Headless / SSH mode: render to PNG file(s) and exit without a GUI.
        if self.save_path is not None:
            self._save_all()
            plt.close(self.fig)
            return

        # Connect keyboard and mouse events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

        self._render()
        plt.show()

    def _load_tensor(self, idx):
        """Lazy-load a single tensor from disk."""
        t = torch.load(self.paths[idx], map_location='cpu', weights_only=True).float()
        # Handle legacy [1, 4, H, W] format
        while t.dim() > 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t

    def _render(self):
        """Draw the current image pair."""
        tensor = self._load_tensor(self.idx)  # [4, H, W]
        name = self.names[self.idx]

        # Preprocessed tensors are saved in [0, 1], but generated maps
        # (e.g. <run>_rgbd_map.pt) are in [-1, 1]. Detect the latter by a
        # negative minimum and rescale into [0, 1] so both display correctly.
        if tensor.min() < -0.01:
            tensor = (tensor + 1.0) / 2.0

        rgb = tensor[:3].permute(1, 2, 0).numpy()    # [H, W, 3]
        depth = tensor[3].numpy()                      # [H, W]

        # Clamp to valid display range
        rgb = np.clip(rgb, 0, 1)
        depth = np.clip(depth, 0, 1)

        cmap_name = DEPTH_CMAPS[self.cmap_idx % len(DEPTH_CMAPS)]

        # Stats annotations
        rgb_stats = (
            f'min={rgb.min():.3f}  max={rgb.max():.3f}\n'
            f'mean={rgb.mean():.3f}  std={rgb.std():.3f}'
        )
        depth_stats = (
            f'min={depth.min():.3f}  max={depth.max():.3f}\n'
            f'mean={depth.mean():.3f}  std={depth.std():.3f}'
        )

        if self.im_rgb is None:
            # ── Initial setup (first frame only) ──
            self.im_rgb = self.ax_rgb.imshow(rgb)
            self.ax_rgb.set_title('RGB', fontsize=12, fontweight='bold', color='#e0e0e0', pad=8)
            self.ax_rgb.axis('off')
            self.ax_rgb.set_facecolor('#1a1a2e')

            self.txt_rgb = self.ax_rgb.text(
                0.02, 0.02, rgb_stats, transform=self.ax_rgb.transAxes,
                fontsize=7, color='#cccccc', verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000', alpha=0.6)
            )

            # ── Depth panel ──
            self.im_depth = self.ax_depth.imshow(depth, cmap=cmap_name, vmin=0, vmax=1)
            self.ax_depth.set_title(
                f'Depth  ({cmap_name})', fontsize=12, fontweight='bold', color='#e0e0e0', pad=8
            )
            self.ax_depth.axis('off')
            self.ax_depth.set_facecolor('#1a1a2e')

            self.txt_depth = self.ax_depth.text(
                0.02, 0.02, depth_stats, transform=self.ax_depth.transAxes,
                fontsize=7, color='#cccccc', verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000', alpha=0.6)
            )

            # ── Colorbar ──
            self.cbar = self.fig.colorbar(
                self.im_depth, ax=self.ax_depth, fraction=0.046, pad=0.04
            )
            self.cbar.ax.tick_params(labelsize=7, colors='#aaaaaa')
        else:
            # ── Fast update of existing artists (prevents colorbar.remove bugs) ──
            self.im_rgb.set_data(rgb)
            self.txt_rgb.set_text(rgb_stats)

            self.im_depth.set_data(depth)
            self.im_depth.set_cmap(cmap_name)
            self.ax_depth.set_title(
                f'Depth  ({cmap_name})', fontsize=12, fontweight='bold', color='#e0e0e0', pad=8
            )
            self.txt_depth.set_text(depth_stats)
            self.cbar.draw_all()  # Force update colorbar mapping

        # ── Supertitle ──
        h, w = tensor.shape[1], tensor.shape[2]
        self.suptitle.set_text(
            f'{name}   [{self.idx + 1}/{self.count}]   •   {w}×{h}   •   tensor [4, {h}, {w}]'
        )

        self.fig.canvas.draw_idle()

    def _save_all(self):
        """Headless mode: render every tensor to a PNG instead of opening a GUI."""
        out = Path(self.save_path)
        # A .png target holds a single image; anything else is treated as a
        # directory that receives one <name>_view.png per tensor.
        single_file_target = out.suffix.lower() == '.png' and self.count == 1
        if single_file_target:
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out.mkdir(parents=True, exist_ok=True)

        for i in range(self.count):
            self.idx = i
            self._render()
            dest = out if single_file_target else out / f"{self.names[i]}_view.png"
            self.fig.savefig(
                dest, dpi=150, facecolor=self.fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3
            )
            print(f"Saved: {dest}")

    def _on_key(self, event):
        if event.key in ('right', 'd'):
            self.idx = (self.idx + 1) % self.count
            self._render()
        elif event.key in ('left', 'a'):
            self.idx = (self.idx - 1) % self.count
            self._render()
        elif event.key == 'home':
            self.idx = 0
            self._render()
        elif event.key == 'end':
            self.idx = self.count - 1
            self._render()
        elif event.key == 'c':
            self.cmap_idx = (self.cmap_idx + 1) % len(DEPTH_CMAPS)
            self._render()
        elif event.key == 's':
            save_path = self.output_dir / f'{self.names[self.idx]}_comparison.png'
            self.fig.savefig(save_path, dpi=150, facecolor=self.fig.get_facecolor(),
                            bbox_inches='tight', pad_inches=0.3)
            print(f'Saved: {save_path}')
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)

    def _on_scroll(self, event):
        if event.button == 'up':
            # Scroll up -> previous image
            self.idx = (self.idx - 1) % self.count
            self._render()
        elif event.button == 'down':
            # Scroll down -> next image
            self.idx = (self.idx + 1) % self.count
            self._render()


# ─── Loading helpers ──────────────────────────────────────────────────

def load_preprocessed(data_dir: str):
    """Return paths and names for preprocessed RGBD tensors (lazy-loaded by viewer)."""
    rgbd_dir = Path(data_dir) / "rgbd"
    if not rgbd_dir.exists():
        print(f"Error: No 'rgbd' folder found in {data_dir}")
        sys.exit(1)

    pt_files = sorted(rgbd_dir.glob("*.pt"))
    if not pt_files:
        print(f"Error: No .pt files found in {rgbd_dir}")
        sys.exit(1)

    paths = [str(f) for f in pt_files]
    names = [f.stem.replace('_rgbd', '') for f in pt_files]

    print(f"Found {len(paths)} preprocessed RGBD tensors in {rgbd_dir} (lazy-loaded)")
    return paths, names


def load_single_file(file_path: str):
    """Return path/name for a single RGBD .pt tensor (lazy-loaded by viewer)."""
    p = Path(file_path)
    if not p.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    name = p.stem.replace('_rgbd', '').replace('_rgbd_map', '')
    print(f"Loading single RGBD tensor: {p}")
    return [str(p)], [name]


def load_raw(raw_dir: str, device: str = 'cpu'):
    """Process raw images with depth estimation and cache to temp .pt files for lazy viewing."""
    import glob
    import tempfile
    from dreamsea.data_preprocessing import DataPreprocessor

    raw_path = Path(raw_dir)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(str(raw_path / ext)))
        image_paths.extend(glob.glob(str(raw_path / ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"Error: No images found in {raw_dir}")
        sys.exit(1)

    # Create a temp directory to cache processed tensors
    cache_dir = Path(tempfile.mkdtemp(prefix='dreamsea_viewer_'))
    print(f"Found {len(image_paths)} images. Running depth estimation (cache: {cache_dir})...")
    preprocessor = DataPreprocessor(device=device)

    paths = []
    names = []
    for i, img_path in enumerate(image_paths):
        try:
            t = preprocessor.process_rgb_to_rgbd(img_path).cpu().float()
            while t.dim() > 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            stem = Path(img_path).stem
            cache_path = cache_dir / f"{stem}_rgbd.pt"
            torch.save(t, cache_path)
            paths.append(str(cache_path))
            names.append(stem)
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

    print(f"Processed {len(paths)} RGBD tensors (cached for lazy viewing)")
    return paths, names


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DreamSea RGBD Viewer — Compare RGB vs Depth side by side."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-d", "--data_dir", type=str,
        help="Path to preprocessed output directory (containing 'rgbd/' folder with .pt files)"
    )
    group.add_argument(
        "-f", "--file", type=str,
        help="Path to a single RGBD .pt tensor ([4, H, W]) to view, e.g. an "
             "<name>_rgbd.pt or a generated <run>_rgbd_map.pt"
    )
    group.add_argument(
        "-r", "--raw_dir", type=str,
        help="Path to raw RGB images directory (will run depth estimation live)"
    )
    parser.add_argument(
        "-e", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for live depth estimation (only used with --raw_dir)"
    )
    parser.add_argument(
        "-o", "--save", type=str, default=None,
        help="Render to PNG instead of opening a GUI (for headless / SSH use). "
             "Give a .png path for a single tensor, or a directory for many. "
             "If omitted and no display is detected, PNGs are written to the cwd."
    )

    args = parser.parse_args()

    # Decide GUI vs headless. Over plain SSH there is no display (no DISPLAY on
    # Linux/macOS); render to PNG there. An explicit --save also forces headless.
    has_display = sys.platform == "win32" or bool(os.environ.get("DISPLAY"))
    headless = bool(args.save) or not has_display

    save_path = args.save
    if headless and save_path is None:
        save_path = "."  # write <name>_view.png into the current directory
        print("No display detected (headless/SSH) — rendering to PNG in the current "
              "directory instead of opening a window. Use --save to choose a path.")

    if not headless:
        # A display is available: switch from the safe Agg default to an
        # interactive backend. force=True is fine here — no figure exists yet.
        matplotlib.use("TkAgg", force=True)

    if args.data_dir:
        paths, names = load_preprocessed(args.data_dir)
    elif args.file:
        paths, names = load_single_file(args.file)
    else:
        paths, names = load_raw(args.raw_dir, args.device)

    if not paths:
        print("No images to display.")
        sys.exit(1)

    RGBDViewer(paths, names, save_path=save_path)
