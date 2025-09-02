import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import get_point_group


def add_indices_to_dataframe(df, x_col="x", y_col="y", round_decimals=6):
    """
    Given a DataFrame with float x,y scan coordinates,
    compute integer indices (x_indice, y_indice) by ranking unique coordinate values,
    insert them after x,y columns, and return a new DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing x,y coordinates
        x_col (str): Column name for x coordinates
        y_col (str): Column name for y coordinates
        round_decimals (int): Rounding precision before ranking floats

    Returns:
        df_out (pd.DataFrame): DataFrame with new x_indice, y_indice
    """
    # Round coords to mitigate float jitter
    x_rounded = df[x_col].round(round_decimals)
    y_rounded = df[y_col].round(round_decimals)

    # Unique sorted levels → index mapping
    x_levels = np.sort(x_rounded.unique())
    y_levels = np.sort(y_rounded.unique())
    x_map = {v: i for i, v in enumerate(x_levels)}
    y_map = {v: i for i, v in enumerate(y_levels)}

    # Apply mapping
    df = df.copy()
    df["x_indice"] = x_rounded.map(x_map).astype(int)
    df["y_indice"] = y_rounded.map(y_map).astype(int)

    # Reorder columns: put x_indice after x, y_indice after y
    cols = []
    for c in df.columns:
        cols.append(c)
        if c == x_col:
            cols.append("x_indice")
        if c == y_col:
            cols.append("y_indice")
    # Ensure no duplicates
    cols = [c for i, c in enumerate(cols) if c not in cols[:i]]
    df_out = df[cols]

    print("x range:", df_out['x_indice'].min(), "→", df_out['x_indice'].max(),
          "n unique:", df_out['x_indice'].nunique())
    print("y range:", df_out['y_indice'].min(), "→", df_out['y_indice'].max(),
          "n unique:", df_out['y_indice'].nunique())
    
    return df_out


def plot_phase_map_roi(
    df,
    roi_x=None,                  # tuple (xmin, xmax) in index space, inclusive
    roi_y=None,                  # tuple (ymin, ymax) in index space, inclusive
    phase_colors=None,           # dict {phase_id: [r,g,b]} in 0..1
    phase_labels=None,           # dict {phase_id: "name"}
    figsize=(12, 9),
    show_legend=True,
    save_path=None               # e.g. "phase_map_roi.svg" / ".pdf" / ".png"
):
    """
    Plot a phase map from integer-indexed EBSD dataframe, optionally cropped to a ROI.

    The dataframe must contain at least: 'x_indice', 'y_indice', 'phase_id'.
    If 'eul1','eul2','eul3' exist, Euler angle grids are also constructed and returned.

    Args
    ----
    df : pandas.DataFrame
        Must include columns: x_indice (int), y_indice (int), phase_id (int).
        Optional columns: eul1, eul2, eul3 (floats).
    roi_x : tuple(int, int) or None
        Inclusive x-index range (xmin, xmax). If None, uses full [0 .. max_x].
    roi_y : tuple(int, int) or None
        Inclusive y-index range (ymin, ymax). If None, uses full [0 .. max_y].
    phase_colors : dict or None
        Mapping {phase_id: [r,g,b]} in 0..1. Unspecified phase_ids default to light gray.
    phase_labels : dict or None
        Mapping {phase_id: "label"} for legend. Unspecified get str(phase_id).
    figsize : tuple
        Matplotlib figure size.
    show_legend : bool
        Whether to draw the legend.
    save_path : str or None
        If provided, saves the figure (supports vector formats like .svg/.pdf).
    """
    if not {"x_indice", "y_indice", "phase_id"}.issubset(df.columns):
        raise ValueError("df must contain 'x_indice', 'y_indice', 'phase_id' columns.")

    x_idx = df["x_indice"].to_numpy(dtype=int)
    y_idx = df["y_indice"].to_numpy(dtype=int)
    pid   = df["phase_id"].to_numpy()

    max_x = int(x_idx.max())
    max_y = int(y_idx.max())

    # Full grid
    phase_grid = np.full((max_y + 1, max_x + 1), fill_value=-1, dtype=int)
    phase_grid[y_idx, x_idx] = pid

    # ROI handling
    if roi_x is None:
        roi_x = (0, max_x)
    if roi_y is None:
        roi_y = (0, max_y)

    xmin, xmax = map(int, roi_x)
    ymin, ymax = map(int, roi_y)
    xmin = max(0, xmin); xmax = min(max_x, xmax)
    ymin = max(0, ymin); ymax = min(max_y, ymax)
    if xmin > xmax or ymin > ymax:
        raise ValueError("Invalid ROI ranges after clipping to grid bounds.")

    roi_phase_id = phase_grid[ymin:ymax, xmin:xmax]

    # Colors
    present_pids = np.unique(roi_phase_id)
    if phase_colors is None:
        phase_colors = {-1: [1, 1, 1]}  # white for not indexed
    for p in present_pids:
        if p not in phase_colors:
            phase_colors[p] = [0.8, 0.8, 0.8]

    # Build RGB image
    rgb = np.zeros(roi_phase_id.shape + (3,), dtype=float)
    for p in present_pids:
        rgb[roi_phase_id == p] = phase_colors[p]

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(rgb, origin="upper")
    plt.axis("off")

    if show_legend:
        patches = []
        for p in present_pids:
            label = str(p)
            if phase_labels and (p in phase_labels):
                label = phase_labels[p]
            patches.append(mpatches.Patch(color=phase_colors[p], label=label))
        if patches:
            plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",borderaxespad=0, fontsize=10, frameon=True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def mark_phase_boundaries_and_plot(
    df: pd.DataFrame,
    roi_xrange=None,          # e.g. (xmin, xmax), inclusive indices; None = full range
    roi_yrange=None,          # e.g. (ymin, ymax), inclusive indices; None = full range
    phase_colors=None,        # dict {phase_id: [r,g,b]} in 0..1
    phase_labels=None,        # dict {phase_id: "label"} for legend. Unspecified IDs → str(phase_id)
    figsize=(10, 8),          # figure size (width, height)
    show_legend=True
):
    """
    Add boundary masks to df and plot a phase map (optionally within an ROI).

    New columns added to df:
      - 'phase_boundary': True if among the 8-neighborhood there are >= 2 distinct non-(-1) phases.
      - 'points_on_the_border': True if (current_phase != -1), at least one neighbor == -1,
                                and all non-(-1) neighbors equal current_phase.

    Args:
        df (pd.DataFrame): Must contain columns ['x_indice','y_indice','phase_id'].
        roi_xrange (tuple[int,int] | None): (xmin, xmax) inclusive indices to display; None = full.
        roi_yrange (tuple[int,int] | None): (ymin, ymax) inclusive indices to display; None = full.
        phase_colors (dict|None): {phase_id: [r,g,b]} (0..1). If None, a small default is used.
        phase_labels (dict|None): {phase_id: "label"} for legend. Unspecified IDs → str(phase_id).
        figsize (tuple[float,float]): Figure size in inches (width, height).
        show_legend (bool): Whether to draw a legend.

    Returns:
        pd.DataFrame: same df with two new boolean columns.
    """
    required = {"x_indice", "y_indice", "phase_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    # ---- build dense grid (rows = y, cols = x); fill missing with -1 ----
    xs = df["x_indice"].to_numpy(int)
    ys = df["y_indice"].to_numpy(int)
    phases = df["phase_id"].to_numpy(int)

    nx = int(xs.max()) + 1
    ny = int(ys.max()) + 1

    phase_grid = np.full((ny, nx), -1, dtype=int)
    phase_grid[ys, xs] = phases  # fill known points

    # ---- compute masks over 8-neighborhood ----
    phase_boundary = np.zeros_like(phase_grid, dtype=bool)
    points_on_the_border = np.zeros_like(phase_grid, dtype=bool)

    # 8-connected neighbor offsets
    nbrs = [(-1,-1), (-1,0), (-1,1),
            ( 0,-1),         ( 0,1),
            ( 1,-1), ( 1,0), ( 1,1)]

    for i in range(ny):
        for j in range(nx):
            c = phase_grid[i, j]
            neighbor_vals = []
            has_unindexed_neighbor = False

            for di, dj in nbrs:
                ii, jj = i + di, j + dj
                if 0 <= ii < ny and 0 <= jj < nx:
                    v = phase_grid[ii, jj]
                    neighbor_vals.append(v)
                    if v == -1:
                        has_unindexed_neighbor = True

            # distinct non-(-1) phases around
            nonneg = [v for v in neighbor_vals if v != -1]
            distinct_nonneg = set(nonneg)

            # (1) boundary if ≥2 distinct non-(-1) phases around
            if len(distinct_nonneg) >= 2:
                phase_boundary[i, j] = True

            # (2) points_on_the_border: has at least one -1 neighbor,
            #     current phase is not -1, and all non-(-1) neighbors equal current phase.
            if c != -1 and has_unindexed_neighbor:
                if len(distinct_nonneg) == 0:
                    # neighbors are all -1 → treat as on-the-border too
                    points_on_the_border[i, j] = True
                else:
                    if len(distinct_nonneg) == 1 and (c in distinct_nonneg):
                        points_on_the_border[i, j] = True

    # ---- write back to df (index by y,x) ----
    df = df.copy()
    df["phase_boundary"] = phase_boundary[ys, xs]
    df["points_on_the_border"] = points_on_the_border[ys, xs]

    # ---- plotting ----
    if phase_colors is None:
        phase_colors = {
            -1: [1, 1, 1],        # unindexed -> white
             1: [0.94, 0.5, 0.5], # Iron bcc (old)
             3: [0.85, 0.65, 0.13], # Hematite
             4: [0.53, 0.81, 0.98], # Magnetite
             5: [0.0, 0.0, 0.55], # Wuestite
        }

    # ROI selection (indices are inclusive)
    if roi_xrange is None:
        xmin, xmax = 0, nx - 1
    else:
        xmin, xmax = int(roi_xrange[0]), int(roi_xrange[1])
        xmin = max(0, xmin); xmax = min(nx-1, xmax)

    if roi_yrange is None:
        ymin, ymax = 0, ny - 1
    else:
        ymin, ymax = int(roi_yrange[0]), int(roi_yrange[1])
        ymin = max(0, ymin); ymax = min(ny-1, ymax)

    roi_phase = phase_grid[ymin:ymax, xmin:xmax].copy()
    roi_boundary = phase_boundary[ymin:ymax, xmin:xmax]
    roi_borderpts = points_on_the_border[ymin:ymax, xmin:xmax]

    H, W = roi_phase.shape
    rgb = np.zeros((H, W, 3), dtype=float)
    for pid, rgb_val in phase_colors.items():
        mask = (roi_phase == pid)
        rgb[mask] = rgb_val

    # overlay: points_on_the_border (light gray), then boundaries (black) on top
    rgb[roi_borderpts] = [0.75, 0.75, 0.75]  # light gray
    rgb[roi_boundary] = [0.0, 0.0, 0.0]      # black

    plt.figure(figsize=figsize)
    plt.imshow(rgb, origin="upper")
    plt.axis("off")

    if show_legend:
        patches = []
        for pid, color in phase_colors.items():
            label = phase_labels[pid] if (phase_labels and pid in phase_labels) else str(pid)
            patches.append(mpatches.Patch(color=color, label=label))
        patches += [
            mpatches.Patch(color=[0.75, 0.75, 0.75], label="points_on_the_border"),
            mpatches.Patch(color=[0.0, 0.0, 0.0], label="phase_boundary"),
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=10,
            frameon=True
        )


    plt.show()

    return df


def compute_kam_gb_from_df(
    df,
    phase_symmetry_dict,
    ny,
    nx,
    kernel_radius: int = 1,
    kam_tolerance: float = 5.0,          # neighbors with mis < this (deg) contribute to KAM
    gb_thresholds=(1.0, 2.0, 5.0, 10.0), # misorientation (deg) above which a pixel is flagged as GB
    include_phase_boundaries: bool = True,
):
    """
    Compute KAM (kernel average misorientation), per-pixel maximum misorientation,
    and grain-boundary maps at one or more thresholds, on a regular ny×nx grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        - 'phase_id'   : int phase identifier (use -1 for unindexed/background)
        - 'eul1','eul2','eul3' : Euler angles (degrees) per pixel.
        The data must be ordered such that `len(df) == ny * nx`, and row-major
        reshaping (ny, nx, ...) matches your grid layout (y rows, x columns).
    phase_symmetry_dict : dict[int, orix.quaternion.symmetry.Symmetry]
        Mapping {phase_id: Symmetry}. If a phase_id is missing, cubic (432) is used.
    ny, nx : int
        Grid height (rows, y) and width (cols, x).
    kernel_radius : int, default 1
        Neighborhood radius. When set to 1, this function uses 4-neighbors
        (up, down, left, right). For radius > 1, a square neighborhood is used.
    kam_tolerance : float, default 5.0
        Only neighbor misorientations <= this value (degrees) are averaged for KAM.
    gb_thresholds : iterable of float, default (1.0, 2.0, 5.0, 10.0)
        One or more GB thresholds (degrees). A pixel is marked True in a GB map
        if its *maximum* neighbor misorientation exceeds the given threshold.
    include_phase_boundaries : bool, default True
        If True, any neighbor belonging to a different phase causes the center
        pixel to be flagged as GB for *all* thresholds.

    Returns
    -------
    kam_map : np.ndarray, shape (ny, nx)
        KAM value per pixel (NaN where not computed).
    max_mis_map : np.ndarray, shape (ny, nx)
        Maximum neighbor misorientation (degrees) per pixel (NaN where missing).
    gb_maps : dict[float, np.ndarray]
        For each threshold T in `gb_thresholds`, a boolean array (ny, nx) where
        True marks grain-boundary pixels (max misorientation > T), optionally
        also including phase boundaries if `include_phase_boundaries=True`.
    result_df : pandas.DataFrame
        Copy of `df` plus:
        - 'KAM'
        - 'max_misorientation'
        - one boolean column per threshold, named 'GB_<T>' (e.g., 'GB_5' or 'GB_2.5').

    Notes
    -----
    * Background/unindexed pixels should have phase_id == -1; they are skipped.
    * Euler-to-rotation conversion uses `orix.quaternion.Rotation.from_euler`
      with degrees=True. Adjust if your convention differs.
    """
    total = len(df)
    if ny * nx != total:
        raise ValueError(f"Grid {ny}x{nx}={ny*nx}, but df has {total} rows.")

    # Prepare outputs
    kam_map = np.full((ny, nx), np.nan, dtype=float)
    max_mis_map = np.full((ny, nx), np.nan, dtype=float)
    gb_maps = {float(T): np.zeros((ny, nx), dtype=bool) for T in gb_thresholds}

    # Reshape core columns
    phase_2d = df["phase_id"].to_numpy().reshape(ny, nx)
    eul_2d = df[["eul1", "eul2", "eul3"]].to_numpy().reshape(ny, nx, 3)

    background_id = -1

    # Build Rotation objects (None for NaNs)
    rotations_2d = np.empty((ny, nx), dtype=object)
    for i in range(ny):
        for j in range(nx):
            ang = eul_2d[i, j]
            if not np.isnan(ang).any():
                # Adjust convention if needed (default here assumes degrees=True)
                rotations_2d[i, j] = Rotation.from_euler(ang, degrees=True)
            else:
                rotations_2d[i, j] = None

    # Neighborhood offsets
    offsets = []
    for di in range(-kernel_radius, kernel_radius + 1):
        for dj in range(-kernel_radius, kernel_radius + 1):
            if di == 0 and dj == 0:
                continue
            if kernel_radius == 1 and (abs(di) + abs(dj) != 1):
                # 4-neighbors when radius=1
                continue
            offsets.append((di, dj))

    # Main loop
    for i in range(ny):
        for j in range(nx):
            phase_c = phase_2d[i, j]
            rot_c = rotations_2d[i, j]
            if rot_c is None or phase_c == background_id:
                continue

            sym_c = phase_symmetry_dict.get(phase_c)
            ori_c = Orientation(rot_c, symmetry=sym_c)

            mis_list = []
            has_phase_boundary = False

            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if not (0 <= ni < ny and 0 <= nj < nx):
                    continue

                phase_n = phase_2d[ni, nj]
                rot_n = rotations_2d[ni, nj]
                if rot_n is None or phase_n == background_id:
                    # treat as missing / void — no misorientation added
                    continue

                if phase_n != phase_c:
                    has_phase_boundary = True
                    continue

                # same phase → compute misorientation
                sym_n = phase_symmetry_dict.get(phase_n)
                ori_n = Orientation(rot_n, symmetry=sym_n)
                mis = ori_c.angle_with(ori_n, degrees=True)
                # angle_with can return array-like; take scalar
                mis = float(np.atleast_1d(mis)[0])
                mis_list.append(mis)

            # KAM
            if mis_list:
                arr = np.asarray(mis_list, dtype=float)
                valid = arr[arr <= kam_tolerance]
                if valid.size > 0:
                    kam_map[i, j] = float(valid.mean())
                max_mis_map[i, j] = float(arr.max())

                # GB masks from max misorientation
                max_mis = float(arr.max())
                for T in gb_maps:
                    if max_mis > T:
                        gb_maps[T][i, j] = True

            # Optionally include phase boundaries in GB maps
            if include_phase_boundaries and has_phase_boundary:
                for T in gb_maps:
                    gb_maps[T][i, j] = True

    # Assemble result DataFrame
    result_df = df.copy()
    result_df["KAM"] = kam_map.ravel()
    result_df["max_misorientation"] = max_mis_map.ravel()
    for T, gmask in gb_maps.items():
        # Clean column name (GB_1 or GB_2.5)
        col = f"GB_{int(T)}" if float(T).is_integer() else f"GB_{T}"
        result_df[col] = gmask.ravel()

    return kam_map, max_mis_map, gb_maps, result_df


def plot_kam_with_overlays(
    df,
    ny: int,
    nx: int,
    *,
    roi_xrange=None,          # e.g. (xmin, xmax) in index space; None = full
    roi_yrange=None,          # e.g. (ymin, ymax) in index space; None = full
    overlays=("phase_boundary", "points_on_the_border",
              "GB_0.1", "GB_0.2", "GB_0.5", "GB_1", "GB_2", "GB_5", "GB_10"),
    overlay_styles=None,      # optional dict to customize colors/linestyles
    fill_overlays=False,      # if True, lightly fill masks beneath contours
    figsize=(12, 8),
    vmax_percentile=95,       # robust upper limit for KAM color scale
    legend=True
):
    """
    Plot a KAM heatmap with optional boundary/GB overlays and ROI cropping.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'KAM' column of length ny*nx (row-major: reshape(ny, nx)).
        Optional overlay columns include:
          - 'phase_boundary' (bool)
          - 'points_on_the_border' (bool)
          - 'GB_1', 'GB_2', 'GB_5', 'GB_10' (bool) or any 'GB_*' column.
    ny, nx : int
        Grid dimensions (rows, cols) to reshape flat columns into 2D.
    roi_xrange : tuple[int, int] or None
        Inclusive (xmin, xmax) in x-index (column axis). None for full width.
    roi_yrange : tuple[int, int] or None
        Inclusive (ymin, ymax) in y-index (row axis). None for full height.
    overlays : iterable[str]
        Column names to overlay as contours (drawn in order).
    overlay_styles : dict or None
        Mapping from overlay name -> style dict:
        {
          "color": str,      # contour color
          "linestyle": str,  # e.g. '-', '--', ':'
          "linewidth": float,
          "alpha": float,    # for optional fills
        }
        Any missing keys use sensible defaults (see code).
    fill_overlays : bool
        If True, lightly fills each overlay area (semi-transparent) in addition
        to drawing a contour. This helps see overlaps without hiding KAM.
    figsize : tuple
        Figure size in inches.
    vmax_percentile : float
        Upper clamp for KAM colormap (percentile of non-NaN KAM).
    legend : bool
        If True, draws a legend describing overlays.

    Notes
    -----
    - Overlap clarity: contours are drawn with distinct colors/linestyles,
      so even overlapping regions remain distinguishable. Optional
      translucent fills can be enabled via `fill_overlays=True`.
    - ROI: indices are inclusive; out-of-bounds are clipped safely.
    """
    # --------- build KAM 2D ----------
    if "KAM" not in df.columns:
        raise ValueError("DataFrame must contain a 'KAM' column.")
    kam_2d = df["KAM"].to_numpy().reshape(ny, nx)

    # Custom warm KAM colormap (your palette)
    kam_cmap = LinearSegmentedColormap.from_list(
        'kam', [
            "#fff5f5", "#ffdada", "#ffbcbc", "#ff9b9b",
            "#ff7b7b", "#ff5c5c", "#e64545", "#cc3030", "#b21a1a"
        ]
    )

    # --------- ROI slicing ----------
    x0, x1 = 0, nx - 1
    y0, y1 = 0, ny - 1
    if roi_xrange is not None:
        x0 = max(0, int(roi_xrange[0]))
        x1 = min(nx - 1, int(roi_xrange[1]))
        if x1 < x0:
            raise ValueError("roi_xrange must satisfy xmin <= xmax.")
    if roi_yrange is not None:
        y0 = max(0, int(roi_yrange[0]))
        y1 = min(ny - 1, int(roi_yrange[1]))
        if y1 < y0:
            raise ValueError("roi_yrange must satisfy ymin <= ymax.")

    kam_roi = kam_2d[y0:y1, x0:x1]

    # vmax based on robust percentile across ROI (ignore NaN)
    finite_vals = kam_roi[np.isfinite(kam_roi)]
    if finite_vals.size == 0:
        vmax_val = 1.0
    else:
        vmax_val = np.nanpercentile(finite_vals, vmax_percentile)

    # --------- default overlay styles ----------
    default_styles = {
        "phase_boundary":       dict(color="#111111", linestyle="-",  linewidth=1.2, alpha=0.12),
        "points_on_the_border": dict(color="#6e6e6e", linestyle="--", linewidth=0.9, alpha=0.10),

        "GB_0.1":               dict(color="#17becf", linestyle=":",  linewidth=1.0, alpha=0.10),  # teal dotted
        "GB_0.2":               dict(color="#8c564b", linestyle="-.", linewidth=1.0, alpha=0.10),  # brown dash-dot
        "GB_0.5":               dict(color="#e377c2", linestyle="--", linewidth=1.0, alpha=0.10),  # pink dashed

        "GB_1":                 dict(color="#1f77b4", linestyle="-",  linewidth=1.0, alpha=0.10),  # blue solid
        "GB_2":                 dict(color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.10),  # green dashed
        "GB_5":                 dict(color="#d62728", linestyle="-.", linewidth=1.2, alpha=0.10),  # red dash-dot
        "GB_10":                dict(color="#9467bd", linestyle=":",  linewidth=1.2, alpha=0.10),  # purple dotted
    }
    styles = {**default_styles, **(overlay_styles or {})}

    # --------- prepare overlay masks (ROI-sliced) ----------
    overlay_masks = []
    legend_handles = []
    for name in overlays:
        if name not in df.columns:
            # silently skip missing overlay
            continue
        mask_2d = df[name].to_numpy().reshape(ny, nx)[y0:y1, x0:x1].astype(bool)
        if mask_2d.any():
            st = styles.get(name, default_styles.get(name, dict(color="k", linestyle="-", linewidth=1.0, alpha=0.10)))
            overlay_masks.append((name, mask_2d, st))
            # For legend (line)
            legend_handles.append(
                Line2D([0], [0],
                       color=st["color"], linestyle=st["linestyle"],
                       linewidth=st["linewidth"], label=name)
            )

    # --------- plot ----------
    plt.figure(figsize=figsize)
    im = plt.imshow(
        kam_roi,
        cmap=kam_cmap,
        vmin=0.0,
        vmax=vmax_val,
        origin="upper",
        interpolation="nearest"
    )
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('KAM (degrees)', fontsize=11)

    # Optional translucent fills (under contours) to visualize overlaps
    if fill_overlays:
        for _, mask, st in overlay_masks:
            # Build an RGBA layer where mask True uses overlay color with alpha
            # Create a 3D array for colors
            overlay_rgb = np.zeros((*mask.shape, 4), dtype=float)
            # Convert hex to rgb if needed
            col = st["color"]
            rgb = np.array(plt.matplotlib.colors.to_rgb(col))
            overlay_rgb[..., :3] = rgb
            overlay_rgb[..., 3] = st.get("alpha", 0.10) * mask.astype(float)
            plt.imshow(overlay_rgb, origin="upper", interpolation="nearest")

    # Contour edges for clarity (draw after fills)
    for name, mask, st in overlay_masks:
        # Draw a single contour around the True-region(s)
        # Convert to float for contour (levels=[0.5] between 0/1)
        plt.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[st["color"]],
            linestyles=[st["linestyle"]],
            linewidths=[st["linewidth"]],
            alpha=1.0
        )

    # Labels & legend
    plt.title("KAM Map" + (f"  ROI x[{x0},{x1}] y[{y0},{y1}]" if (roi_xrange or roi_yrange) else ""), fontsize=13)
    plt.xlabel("x index")
    plt.ylabel("y index")

    if legend and legend_handles:
        plt.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.show()


