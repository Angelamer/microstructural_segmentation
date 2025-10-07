
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Literal, List

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.neighbors import NearestNeighbors
except Exception as e:
    raise RuntimeError("scikit-learn is required for clustering. Please ensure it is installed.") from e

# hdbscan is optional
try:
    import hdbscan as _hdbscan
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False


# -----------------------------
# Utilities
# -----------------------------
def _is_full_grid(xs, ys):
    xs = np.asarray(xs); ys = np.asarray(ys)
    Xuniq = np.unique(xs); Yuniq = np.unique(ys)
    all_pairs = {(int(x), int(y)) for x in Xuniq for y in Yuniq}
    coords = {(int(x), int(y)) for x, y in zip(xs, ys)}
    full = (coords == all_pairs) and (len(coords) == len(Xuniq) * len(Yuniq))
    return full, np.sort(Xuniq), np.sort(Yuniq)


def _render_field(ax, xs, ys, rgb_or_scalar, title: str, cmap="RdBu", add_colorbar=True):
    xs = np.asarray(xs); ys = np.asarray(ys)
    V = np.asarray(rgb_or_scalar)

    full, Xsorted, Ysorted = _is_full_grid(xs, ys)
    if full:
        H, W = len(Ysorted), len(Xsorted)
        if V.ndim == 1:  # scalar heatmap
            grid = np.full((H, W), np.nan, dtype=float)
        elif V.ndim == 2 and V.shape[1] == 3:  # RGB grid
            grid = np.full((H, W, 3), np.nan, dtype=float)
        else:
            raise ValueError("Unexpected value shape for grid: expected (N,) or (N,3).")
        x_pos = {x: i for i, x in enumerate(Xsorted)}
        y_pos = {y: i for i, y in enumerate(Ysorted)}
        if V.ndim == 1:
            for x, y, val in zip(xs, ys, V):
                grid[y_pos[int(y)], x_pos[int(x)]] = val
            im = ax.imshow(np.ma.masked_invalid(grid), origin="upper", cmap=cmap, aspect="auto")
            if add_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            for (x, y), rgb in zip(zip(xs, ys), V):
                grid[y_pos[int(y)], x_pos[int(x)]] = rgb
            ax.imshow(grid, origin="upper", aspect="auto")
    else:
        if V.ndim == 1:
            sc = ax.scatter(xs, ys, c=V, s=6, cmap=cmap, linewidths=0)
            if add_colorbar:
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.scatter(xs, ys, c=V, s=6, linewidths=0)
        ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')


def _roi_mask_from_ranges(xs, ys, x_range=None, y_range=None):
    xs = np.asarray(xs); ys = np.asarray(ys)
    m = np.ones(xs.shape[0], dtype=bool)
    if x_range is not None:
        xmin, xmax = x_range
        m &= (xs >= xmin) & (xs < xmax)
    if y_range is not None:
        ymin, ymax = y_range
        m &= (ys >= ymin) & (ys < ymax)
    return m


def _labels_to_rgb_shaded(
    latent_roi: np.ndarray,
    labels: np.ndarray,
    saturation: float = 0.85,
    base_colors: Optional[List[str]] = None,
    shade_clip_quantiles: Tuple[float,float] = (0.02, 0.98)
) -> np.ndarray:
    from matplotlib.colors import hsv_to_rgb, to_rgb, rgb_to_hsv

    latent_roi = np.asarray(latent_roi, float).ravel()
    labels = np.asarray(labels, int).ravel()
    n = latent_roi.shape[0]
    ulabels = np.unique(labels[labels >= 0])
    K = len(ulabels)

    # Build label -> hue mapping
    label_to_hue = {}
    if base_colors is not None and len(base_colors) > 0:
        for i, lab in enumerate(sorted(ulabels)):
            rgb = np.array(to_rgb(base_colors[i % len(base_colors)]), dtype=float)
            h, s, v = rgb_to_hsv(rgb.reshape(1,1,3)).reshape(3)
            label_to_hue[lab] = float(h)
    else:
        hue_list = np.linspace(0, 1, max(K,1), endpoint=False)
        for i, lab in enumerate(sorted(ulabels)):
            label_to_hue[lab] = float(hue_list[i % len(hue_list)])

    qlo, qhi = shade_clip_quantiles
    qlo = float(np.clip(qlo, 0.0, 0.5))
    qhi = float(np.clip(qhi, 0.5, 1.0))

    shade = np.zeros(n, dtype=float)
    for lab in ulabels:
        idx = np.where(labels == lab)[0]
        v = latent_roi[idx]
        if v.size == 0 or not np.isfinite(v).any():
            shade[idx] = 0.5
            continue
        lo = np.nanquantile(v, qlo)
        hi = np.nanquantile(v, qhi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(v), np.nanmax(v)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                shade[idx] = 0.5
                continue
        shade[idx] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    hsv = np.zeros((n, 3), dtype=float)
    for i in range(n):
        lab = labels[i]
        if lab < 0:
            hsv[i] = (0.0, 0.0, 0.0)  # noise -> black
        else:
            hsv[i, 0] = label_to_hue[lab]
            hsv[i, 1] = saturation
            hsv[i, 2] = shade[i] * 0.9 + 0.1
    rgb = hsv_to_rgb(hsv)
    return rgb


def estimate_dbscan_eps(latent_roi: np.ndarray, bc_roi: np.ndarray, k: int = 8, quantile: float = 0.98) -> float:
    X = np.column_stack([np.asarray(latent_roi, float).ravel(),
                         np.asarray(bc_roi, float).ravel()])
    good = np.isfinite(X).all(axis=1)
    Xg = X[good]
    if Xg.shape[0] < max(10, k+1):
        raise ValueError("Not enough valid points to estimate eps.")
    nn = NearestNeighbors(n_neighbors=k).fit(Xg)
    dists, _ = nn.kneighbors(Xg)
    kd = dists[:, k-1]
    eps = float(np.quantile(kd, quantile))
    return eps


# -----------------------------
# Clustering methods
# -----------------------------
def cluster_latent_bandcontrast(
    latent_roi: np.ndarray,
    bc_roi: np.ndarray,
    method: Literal["gmm", "kmeans", "dbscan", "hdbscan"] = "gmm",
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 8,
    gmm_criterion: Literal["bic", "aic"] = "bic",
    random_state: int = 0,
    # DBSCAN params
    eps: Optional[float] = None,
    min_samples: int = 10,
    # HDBSCAN params
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    hdbscan_cluster_selection_epsilon: float = 0.0
) -> Tuple[np.ndarray, Dict]:
    X = np.column_stack([np.asarray(latent_roi, float).ravel(),
                         np.asarray(bc_roi, float).ravel()])
    good = np.isfinite(X).all(axis=1)
    Xg = X[good]
    if Xg.shape[0] < 10:
        raise ValueError("Not enough valid points for clustering.")

    labels = -np.ones(X.shape[0], dtype=int)

    if method == "gmm":
        if k is None:
            best_score = np.inf
            best_gmm = None
            best_k = None
            for kk in range(max(2, k_min), max(2, k_max) + 1):
                gm = GaussianMixture(n_components=kk, covariance_type="full", random_state=random_state)
                gm.fit(Xg)
                score = gm.bic(Xg) if gmm_criterion == "bic" else gm.aic(Xg)
                if score < best_score:
                    best_score, best_gmm, best_k = score, gm, kk
            gmm = best_gmm
            k_sel = best_k
            scores = {"criterion": gmm_criterion, "value": float(best_score)}
        else:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state).fit(Xg)
            k_sel = k
            scores = None
        lab_g = gmm.predict(Xg)
        labels[good] = lab_g
        info = {"method": "gmm", "k": int(k_sel), "scores": scores}

    elif method == "kmeans":
        k_sel = 3 if (k is None) else int(k)
        km = KMeans(n_clusters=k_sel, n_init="auto", random_state=random_state).fit(Xg)
        lab_g = km.labels_
        labels[good] = lab_g
        info = {"method": "kmeans", "k": int(k_sel)}

    elif method == "dbscan":
        if eps is None:
            try:
                eps = estimate_dbscan_eps(latent_roi, bc_roi, k=8, quantile=0.98)
            except Exception:
                med = np.median(np.linalg.norm(Xg - Xg.mean(axis=0, keepdims=True), axis=1))
                eps = float(max(med * 0.1, 1e-6))
        db = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(Xg)
        lab_g = db.labels_
        labels[good] = lab_g
        info = {"method": "dbscan", "eps": float(eps), "min_samples": int(min_samples)}

    elif method == "hdbscan":
        if not _HDBSCAN_AVAILABLE:
            raise RuntimeError("hdbscan is not installed. Please `pip install hdbscan`.")
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=int(hdbscan_min_cluster_size),
            min_samples=(None if hdbscan_min_samples is None else int(hdbscan_min_samples)),
            cluster_selection_epsilon=float(hdbscan_cluster_selection_epsilon)
        ).fit(Xg)
        lab_g = clusterer.labels_
        labels[good] = lab_g
        info = {
            "method": "hdbscan",
            "min_cluster_size": int(hdbscan_min_cluster_size),
            "min_samples": (None if hdbscan_min_samples is None else int(hdbscan_min_samples)),
            "cluster_selection_epsilon": float(hdbscan_cluster_selection_epsilon)
        }

    else:
        raise ValueError("method must be 'gmm', 'kmeans', 'dbscan', or 'hdbscan'.")

    return labels, info


# -----------------------------
# Main plotting helper
# -----------------------------
def plot_clustered_latent_vs_bc(
    out_path: str,
    xs_roi: np.ndarray,
    ys_roi: np.ndarray,
    latent_roi: np.ndarray,
    bc_roi: np.ndarray,
    labels: np.ndarray,
    title_left: str = "Cluster-shaded latent map (ROI)",
    title_right: str = "Latent vs BandContrast (clusters)",
    scatter_alpha: float = 0.9,
    cluster_colors: Optional[List[str]] = None,
    shade_clip_quantiles: Tuple[float,float] = (0.02, 0.98)
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    xs = np.asarray(xs_roi); ys = np.asarray(ys_roi)
    latent = np.asarray(latent_roi, float).ravel()
    bc = np.asarray(bc_roi, float).ravel()
    labels = np.asarray(labels, int).ravel()

    # Colors for both panels, using user-defined base colors if provided
    rgb = _labels_to_rgb_shaded(
        latent, labels, saturation=0.9,
        base_colors=cluster_colors,
        shade_clip_quantiles=shade_clip_quantiles
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    _render_field(axes[0], xs, ys, rgb, title_left, add_colorbar=False)

    ulabels = np.unique(labels[labels >= 0])
    for lab in ulabels:
        idx = labels == lab
        axes[1].scatter(latent[idx], bc[idx], s=8, alpha=scatter_alpha, label=f"Cluster {lab}", c=rgb[idx])
    noise_idx = labels < 0
    if np.any(noise_idx):
        axes[1].scatter(latent[noise_idx], bc[noise_idx], s=8, alpha=0.5, label="Noise", c="#999999")

    axes[1].set_xlabel("latent value")
    axes[1].set_ylabel("BandContrast")
    axes[1].set_title(title_right)
    axes[1].legend(loc="best", markerscale=1.2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -----------------------------
# High-level orchestration
# -----------------------------
def cluster_and_plot_from_arrays(
    out_dir: str,
    xs_roi: np.ndarray,
    ys_roi: np.ndarray,
    latent_roi: np.ndarray,
    bc_roi: np.ndarray,
    method: Literal["gmm", "kmeans", "dbscan", "hdbscan"] = "gmm",
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 8,
    gmm_criterion: Literal["bic", "aic"] = "bic",
    random_state: int = 0,
    tag: Optional[str] = None,
    # DBSCAN
    eps: Optional[float] = None,
    min_samples: int = 10,
    # HDBSCAN
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    # Coloring
    cluster_colors: Optional[List[str]] = None,
    shade_clip_quantiles: Tuple[float,float] = (0.02, 0.98)
) -> Dict:
    if len(latent_roi) != len(bc_roi):
        raise ValueError(f"Length mismatch: latent_roi={len(latent_roi)} vs bc_roi={len(bc_roi)}. "
                         "Ensure both refer to the SAME ROI and order.")
    os.makedirs(out_dir, exist_ok=True)
    labels, info = cluster_latent_bandcontrast(
        latent_roi, bc_roi, method=method, k=k, k_min=k_min, k_max=k_max,
        gmm_criterion=gmm_criterion, random_state=random_state,
        eps=eps, min_samples=min_samples,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon
    )

    if tag is None:
        if method in ("gmm","kmeans"):
            tag = f"{info['method']}_k{info.get('k','?')}"
        elif method == "dbscan":
            tag = f"dbscan_eps{info['eps']:.3g}_m{info['min_samples']}"
        else:
            tag = f"hdbscan_mcs{info['min_cluster_size']}"

    fig_path = os.path.join(out_dir, f"cluster_latent_bc__{tag}.png")
    plot_clustered_latent_vs_bc(
        fig_path, xs_roi, ys_roi, latent_roi, bc_roi, labels,
        cluster_colors=cluster_colors, shade_clip_quantiles=shade_clip_quantiles
    )

    assign_path = os.path.join(out_dir, f"cluster_assignments__{tag}.csv")
    df = pd.DataFrame({"x": xs_roi, "y": ys_roi, "latent": latent_roi, "bandcontrast": bc_roi, "cluster": labels})
    df.to_csv(assign_path, index=False)

    report = {
        "method": info["method"],
        **{k:v for k,v in info.items() if k != "method"},
        "n_points": int(len(xs_roi)),
        "figure": fig_path,
        "assignments_csv": assign_path,
        "cluster_colors": cluster_colors
    }
    with open(os.path.join(out_dir, f"cluster_report__{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


# -----------------------------
# Convenience loader for your existing files
# -----------------------------
def demo_from_files(
    out_dir: str,
    coords_csv: str,
    latent_roi_file: str,
    bc_roi_file: Optional[str] = None,
    method: Literal["gmm", "kmeans", "dbscan", "hdbscan"] = "gmm",
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 8,
    gmm_criterion: Literal["bic", "aic"] = "bic",
    random_state: int = 0,
    tag: Optional[str] = None,
    # DBSCAN
    eps: Optional[float] = None,
    min_samples: int = 10,
    # HDBSCAN
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    # Coloring
    cluster_colors: Optional[List[str]] = None,
    shade_clip_quantiles: Tuple[float,float] = (0.02, 0.98),
    # ROI selection from FULL coords
    roi_xrange: Optional[Tuple[int,int]] = None,
    roi_yrange: Optional[Tuple[int,int]] = None
) -> Dict:
    """
    Typical usage with your artifacts:
      - coords_csv: CSV with columns ['x','y'] for the FULL field; may also include 'bandcontrast'.
      - roi_xrange, roi_yrange: if provided, we will select ROI rows from coords_csv.
      - latent_roi_file: ROI vector (.npy or .csv with 'latent' column). Length must equal ROI size.
      - bc_roi_file: optional ROI vector; if None, we will read 'bandcontrast' from coords_csv ROI rows.

    Returns the same dict as `cluster_and_plot_from_arrays`.
    """
    # Load all coords (FULL field)
    cdf = pd.read_csv(coords_csv)
    if not {"x","y"}.issubset(cdf.columns):
        raise ValueError("coords_csv must contain columns 'x' and 'y'.")

    xs_all = cdf["x"].to_numpy(dtype=int)
    ys_all = cdf["y"].to_numpy(dtype=int)
    mask = _roi_mask_from_ranges(xs_all, ys_all, roi_xrange, roi_yrange) if (roi_xrange or roi_yrange) else np.ones_like(xs_all, dtype=bool)

    # ROI coords to drive plotting & optional BC extraction
    xs_roi = xs_all[mask]
    ys_roi = ys_all[mask]

    # latent ROI
    if latent_roi_file.lower().endswith(".npy"):
        latent_roi = np.load(latent_roi_file)
    else:
        ldf = pd.read_csv(latent_roi_file)
        col = "latent" if "latent" in ldf.columns else ldf.columns[0]
        latent_roi = ldf[col].to_numpy(float)

    if len(latent_roi) != xs_roi.shape[0]:
        raise ValueError(
            f"Latent vector length ({len(latent_roi)}) does not match ROI size ({xs_roi.shape[0]}). "
            "Ensure the latent file is computed for the SAME ROI (x,y ranges) and in ROI order."
        )

    # bandcontrast ROI
    if bc_roi_file is None:
        if "bandcontrast" not in cdf.columns:
            raise ValueError("bandcontrast not found in coords_csv – provide bc_roi_file or include it in coords_csv as 'bandcontrast'.")
        bc_full = cdf["bandcontrast"].to_numpy(float)
        bc_roi = bc_full[mask]
    else:
        # allow either ROI-length or FULL-length input; if FULL-length, filter by mask
        if bc_roi_file.lower().endswith(".npy"):
            bc_arr = np.load(bc_roi_file)
        else:
            bdf = pd.read_csv(bc_roi_file)
            col = "bandcontrast" if "bandcontrast" in bdf.columns else bdf.columns[0]
            bc_arr = bdf[col].to_numpy(float)

        if bc_arr.shape[0] == xs_all.shape[0]:
            bc_roi = bc_arr[mask]
        elif bc_arr.shape[0] == xs_roi.shape[0]:
            bc_roi = bc_arr
        else:
            raise ValueError(
                f"bc_roi_file length ({bc_arr.shape[0]}) must match either FULL coords ({xs_all.shape[0]}) or ROI size ({xs_roi.shape[0]})."
            )

    return cluster_and_plot_from_arrays(
        out_dir, xs_roi, ys_roi, latent_roi, bc_roi,
        method=method, k=k, k_min=k_min, k_max=k_max,
        gmm_criterion=gmm_criterion, random_state=random_state, tag=tag,
        eps=eps, min_samples=min_samples,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        cluster_colors=cluster_colors,
        shade_clip_quantiles=shade_clip_quantiles
    )


if __name__ == "__main__":
    
    report = demo_from_files(
    out_dir="out_cluster_latentvsbc_files",
    coords_csv="~/workflow/process_experiment_data/20min_bandcontrast.csv",           #  x,y；optional: bandcontrast
    latent_roi_file="latent_maps_filtered_selected_dim_512_x_350_420_y_70_140/latent_d40_roi_vs_bandcontrast_pearson.npy",
    bc_roi_file=None,                      # if coords_csv contains the bandcontrast, then None
    method="gmm",
    # min_samples=15,
    k=None,
    k_min=2,
    k_max=8,
    roi_xrange=(350,420),
    roi_yrange=(70,140),
    tag="d512_x_350_420_y_70_140_gmm",
    cluster_colors=["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", 
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",],
    )