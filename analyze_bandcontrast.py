#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ---- journal style ----
from apply_journal_style import apply_journal_style  # ensure this file exists next to this script


# =========================
# CSV loader (robust column detection)
# =========================
def _load_xy_bc(csv_path):
    """
    Load a CSV containing columns x, y, bandcontrast.
    Column names are case/space insensitive: e.g. 'X', 'x_position', 'BandContrast'.
    Returns:
      xs, ys, bc : np.ndarray, shape (N,)
    """
    df = pd.read_csv(csv_path)
    # Normalize column names
    norm = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=norm)

    # Try candidate names
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(f"Required column not found among: {cands}. Got columns: {list(df.columns)}")

    cx = pick("x", "x_position", "xpos")
    cy = pick("y", "y_position", "ypos")
    cb = pick("bandcontrast", "band_contrast")

    xs = pd.to_numeric(df[cx], errors="coerce").to_numpy()
    ys = pd.to_numeric(df[cy], errors="coerce").to_numpy()
    bc = pd.to_numeric(df[cb], errors="coerce").to_numpy()

    m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(bc)
    xs, ys, bc = xs[m].astype(int), ys[m].astype(int), bc[m].astype(float)
    return xs, ys, bc
# =========================
# Print top bandcontrast value
# =========================
def print_top_bandcontrast(csv_path, top_pct=20.0):
    """
    Print the bandcontrast values in the top `top_pct` percent.
    """
    df = pd.read_csv(csv_path)
    if "bandcontrast" not in df.columns:
        raise KeyError("CSV must contain a 'bandcontrast' column.")

    values = df["bandcontrast"].dropna().to_numpy()
    thresh = np.percentile(values, 100 - top_pct)  # 例如 top20% -> 80th percentile
    top_vals = values[values >= thresh]
    top_vals_sorted = np.sort(top_vals)[::-1]      # 降序

    print(f"Top {top_pct}% threshold = {thresh:.6g}")
    print(f"Count = {len(top_vals_sorted)} / Total = {len(values)}")
    print("Values:")
    for v in top_vals_sorted:
        print(v)

# =========================
# 1) Histogram (bin width = 10)
# =========================
def save_bandcontrast_hist(csv_path, out_dir, fig_name, bin_width=10, apply_style=True):
    """
    Read CSV and plot bandcontrast histogram (bin width = 10).
    No title, dpi=300. Output filename = fig_name (adds .png if missing).
    """
    if apply_style:
        apply_journal_style()  # Arial-ish, 9/8 pt, dpi=300

    os.makedirs(out_dir, exist_ok=True)
    _, _, bc = _load_xy_bc(csv_path)

    if bc.size == 0:
        raise RuntimeError("No valid bandcontrast values found in CSV.")

    bc_min = math.floor(np.nanmin(bc) / bin_width) * bin_width
    bc_max = math.ceil(np.nanmax(bc) / bin_width) * bin_width
    bins = np.arange(bc_min, bc_max + bin_width, bin_width, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.hist(bc, bins=bins, edgecolor="black")
    # No title
    # Optional: add axis labels if you need them
    ax.set_xlabel("BandContrast")
    ax.set_ylabel("Count")
    fig.tight_layout()

    out_path = os.path.join(out_dir, _ensure_png(fig_name))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================
# 2) Grayscale heatmap + row profile
# =========================
def _assemble_grid(xs, ys, vals):
    """
    Assemble scattered (x,y,val) into a regular grid:
      - Returns grid(H,W), Xsorted, Ysorted.
      - Missing positions filled with NaN.
    """
    Xsorted = np.sort(np.unique(xs))
    Ysorted = np.sort(np.unique(ys))

    W, H = len(Xsorted), len(Ysorted)
    grid = np.full((H, W), np.nan, dtype=float)

    x_pos = {x: i for i, x in enumerate(Xsorted)}
    y_pos = {y: i for i, y in enumerate(Ysorted)}
    for x, y, v in zip(xs, ys, vals):
        grid[y_pos[int(y)], x_pos[int(x)]] = v

    return grid, Xsorted, Ysorted


def save_heatmap_and_row(csv_path, out_dir, fig_name, y_index, apply_style=True, cmap="gray"):
    """
    Save a figure with two subplots (no titles, dpi=300):
      Left: grayscale heatmap of bandcontrast (origin='upper', so (0,0) at top-left),
            draw a red horizontal line at given y_index.
      Right: bandcontrast profile for that row vs x.
    Output filename = fig_name (adds .png if missing).
    """
    if apply_style:
        apply_journal_style()
    os.makedirs(out_dir, exist_ok=True)
    xs, ys, bc = _load_xy_bc(csv_path)

    grid, Xsorted, Ysorted = _assemble_grid(xs, ys, bc)
    H, W = grid.shape

    if not (0 <= y_index < H):
        raise ValueError(f"y_index out of range: 0..{H-1}, got {y_index}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Left: heatmap (origin='upper' => (0,0) top-left)
    im = axes[0].imshow(np.ma.masked_invalid(grid), origin="upper", cmap=cmap, aspect="auto")
    axes[0].axhline(y_index, color="red", linewidth=1.5)
    axes[0].set_xlabel("x (pixel)")
    axes[0].set_ylabel("y (pixel)")  # note: y=0 is top row due to origin='upper'


    # Right: profile curve for that row
    row_vals = grid[y_index, :]  # (W,)
    m = np.isfinite(row_vals)
    axes[1].plot(Xsorted[m], row_vals[m], '-o', markersize=2, linewidth=1)
    axes[1].set_xlabel("x (pixel)")
    axes[1].set_ylabel("BandContrast")
    
    
    fig.tight_layout()
    out_path = os.path.join(out_dir, _ensure_png(fig_name))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================
# Helper
# =========================
def _ensure_png(name: str) -> str:
    """Ensure filename ends with .png."""
    root, ext = os.path.splitext(name)
    return name if ext.lower() == ".png" else root + ".png"


# =========================
# Optional CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BandContrast histogram and heatmap tools.")
    parser.add_argument("--csv", required=True, help="Path to CSV with x,y,bandcontrast")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--hist_name", default="bandcontrast_hist.png", help="Histogram filename")
    parser.add_argument("--heatmap_name", default="bandcontrast_row.png", help="Heatmap+row filename")
    parser.add_argument( "--y_index", type=int, nargs="+", default=[0],
    help="Row index/indices for bandcontrast curves (0-based). Can specify multiple.")
    parser.add_argument("--bin_width", type=int, default=10, help="Bin width for histogram (default 10)")
    parser.add_argument("--top_pct", type=float, default=None,
                        help="If set, print top percentage bandcontrast values (e.g. 10 or 20)")
    args = parser.parse_args()

    if args.top_pct is not None:
        print_top_bandcontrast(args.csv, args.top_pct)
        
    os.makedirs(args.out_dir, exist_ok=True)

    # Apply style once for CLI runs
    apply_journal_style(base_font="DejaVu Sans")
    
    # Save histogram
    save_bandcontrast_hist(args.csv, args.out_dir, args.hist_name, bin_width=args.bin_width)
    # Save heatmap + row profile for each y_index
    for yi in args.y_index:
        fname = args.heatmap_name
        # if multiple y_index -> append number to filename
        if len(args.y_index) > 1:
            root, ext = os.path.splitext(fname)
            fname = f"{root}_y{yi}{ext or '.png'}"
        save_heatmap_and_row(args.csv, args.out_dir, fname, y_index=yi)