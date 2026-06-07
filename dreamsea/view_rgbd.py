"""
DreamSea RGBD Viewer
====================
Interactive viewer to compare RGB images with their estimated depth maps.

Usage:
    # View preprocessed .pt tensors:
    python -m dreamsea.view_rgbd --data_dir path/to/preprocessed_output

    # View raw images (runs depth estimation live):
    python -m dreamsea.view_rgbd --raw_dir path/to/raw_images --device cuda

Controls:
    Left/Right, A/D, or Scroll — Navigate between images
    Home/End                   — Jump to first/last image
    C                          — Toggle depth colormap (magma/viridis/inferno/plasma)
    S                          — Save current side-by-side figure as PNG
    Q or Escape                — Quit
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch


# ─── Color map rotation ──────────────────────────────────────────────
DEPTH_CMAPS = ['magma', 'viridis', 'inferno', 'plasma', 'cividis', 'turbo']


class RGBDViewer:
    """Interactive side-by-side RGB vs Depth viewer with lazy loading."""

    def __init__(self, paths: list, names: list, output_dir: str = None):
        """
        Args:
            paths: List of file paths to [4, H, W] .pt tensors (loaded on demand).
            names: List of display names for each tensor.
            output_dir: Optional directory to save screenshots into.
        """
        self.paths = paths
        self.names = names
        self.count = len(paths)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.idx = 0
        self.cmap_idx = 0

        # ── Set up figure ──
        self.fig = plt.figure(figsize=(14, 6), facecolor='#1a1a2e')
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
        tensor = self._load_tensor(self.idx)  # [4, H, W], values in [0, 1]
        name = self.names[self.idx]

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
        "-r", "--raw_dir", type=str,
        help="Path to raw RGB images directory (will run depth estimation live)"
    )
    parser.add_argument(
        "-e", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for live depth estimation (only used with --raw_dir)"
    )

    args = parser.parse_args()

    if args.data_dir:
        paths, names = load_preprocessed(args.data_dir)
    else:
        paths, names = load_raw(args.raw_dir, args.device)

    if not paths:
        print("No images to display.")
        sys.exit(1)

    RGBDViewer(paths, names)
