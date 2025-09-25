import os, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# -----------------------------
# ROI helpers
# -----------------------------
def roi_mask_from_ranges(xs, ys, x_range=None, y_range=None):
    """
    Return a boolean mask selecting samples with x in [x_min,x_max] and y in [y_min,y_max].
    If x_range or y_range is None, that axis is not filtered.
    """
    xs = np.asarray(xs); ys = np.asarray(ys)
    mask = np.ones(len(xs), dtype=bool)
    if x_range is not None:
        xmin, xmax = x_range
        mask &= (xs >= xmin) & (xs <= xmax)
    if y_range is not None:
        ymin, ymax = y_range
        mask &= (ys >= ymin) & (ys <= ymax)
    return mask

def align_indices_by_coords(src_x, src_y, tgt_coords):
    """
    Build mapping from src coords to target order.
    Returns: src_idx_for_tgt (M,), valid_mask (M,)
    """
    idx_map = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(src_x, src_y))}
    src_idx = np.array([idx_map.get((int(x), int(y)), -1) for x, y in tgt_coords], dtype=int)
    valid = src_idx >= 0
    return src_idx, valid

# -----------------------------
# CSV loaders
# -----------------------------
def load_feature_csv(csv_path, prefix="PCA_"):
    """
    Load (x,y) and feature columns that start with prefix, e.g. 'PCA_' or 'cNMF_'.
    Returns:
      coords : (M,2) int
      feats  : (M,d) float32
      names  : list of feature column names
    """
    df = pd.read_csv(csv_path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain columns 'x' and 'y'.")
    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(feat_cols) == 0:
        raise ValueError(f"No columns start with prefix '{prefix}'.")
    coords = df[["x", "y"]].to_numpy(dtype=int)
    feats = df[feat_cols].to_numpy(dtype=np.float32)
    return coords, feats, feat_cols

def load_elements_csv(csv_path, element_cols=None):
    """
    Load (x,y) and multiple element percentage columns from one CSV (whole field).
    If element_cols is None, auto-detect as all columns except x,y.
    Returns:
      coords : (N,2) int
      elem   : (N,E) float32
      names  : list of element column names
    """
    df = pd.read_csv(csv_path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'x' and 'y'.")
    if element_cols is None:
        element_cols = [c for c in df.columns if c not in ("x", "y")]
    if len(element_cols) == 0:
        raise ValueError("No element columns found.")
    coords = df[["x", "y"]].to_numpy(dtype=int)
    elem = df[element_cols].to_numpy(dtype=np.float32)
    return coords, elem, element_cols

# -----------------------------
# Plotting (publication friendly)
# -----------------------------
def _grid_or_scatter(ax, xs, ys, values, title, cmap="RdBu", s=6):
    """
    Heatmap with top-left at (min x, min y):
      - full grid -> imshow(origin='upper')
      - otherwise -> scatter + invert_yaxis()
    """
    xs = np.asarray(xs); ys = np.asarray(ys); values = np.asarray(values)
    Xuniq = np.unique(xs); Yuniq = np.unique(ys)
    all_pairs = {(int(x), int(y)) for x in Xuniq for y in Yuniq}
    coords = {(int(x), int(y)) for x, y in zip(xs, ys)}
    full = (coords == all_pairs) and (len(coords) == len(Xuniq) * len(Yuniq))

    if full:
        Xsorted = np.sort(Xuniq)
        Ysorted = np.sort(Yuniq)
        H, W = len(Ysorted), len(Xsorted)
        grid = np.zeros((H, W), dtype=float)
        x_pos = {x: i for i, x in enumerate(Xsorted)}
        y_pos = {y: i for i, y in enumerate(Ysorted)}
        for x, y, v in zip(xs, ys, values):
            grid[y_pos[int(y)], x_pos[int(x)]] = v
        im = ax.imshow(grid, origin="upper", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        sc = ax.scatter(xs, ys, c=values, s=s, cmap=cmap, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')

# -----------------------------
# GMM clustering with auto-k
# -----------------------------
def gmm_auto_k(X, k_min=2, k_max=12, criterion="bic", random_state=0, covariance_type="full"):
    """
    Fit GMMs for k=k_min..k_max and pick the model with lowest BIC/AIC.
    Returns: labels (N,), best_k, best_model
    """
    X = np.asarray(X, dtype=np.float32)
    best_score = np.inf
    best_model = None
    best_k = None
    for k in range(k_min, k_max + 1):
        gm = GaussianMixture(
            n_components=k, covariance_type=covariance_type, random_state=random_state
        )
        gm.fit(X)
        score = gm.bic(X) if criterion.lower() == "bic" else gm.aic(X)
        if score < best_score:
            best_score, best_model, best_k = score, gm, k
    labels = best_model.predict(X)
    return labels, best_k, best_model

def gmm_auto_k_1d(values, **kwargs):
    """1D convenience wrapper."""
    v = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return gmm_auto_k(v, **kwargs)

# -----------------------------
# Clustering similarity
# -----------------------------
def clustering_similarity(a, b, metric="ari"):
    return normalized_mutual_info_score(a, b) if metric == "nmi" else adjusted_rand_score(a, b)
def save_full_latent_map(save_dir, tag, xs_all, ys_all, values_full, cmap="RdBu"):
    """
    Save a full-field latent map as PNG (dpi=300) + CSV.
    Orientation: smallest x/y appear in the TOP-LEFT.
    """
    os.makedirs(save_dir, exist_ok=True)
    xs = np.asarray(xs_all, dtype=int)
    ys = np.asarray(ys_all, dtype=int)
    v  = np.asarray(values_full, dtype=np.float32)
    assert v.shape[0] == xs.shape[0] == ys.shape[0], "Length mismatch."

    # Decide grid vs scatter
    Xuniq = np.unique(xs); Yuniq = np.unique(ys)
    all_pairs = {(int(x), int(y)) for x in Xuniq for y in Yuniq}
    coords = {(int(x), int(y)) for x, y in zip(xs, ys)}
    full_grid = (coords == all_pairs) and (len(coords) == len(Xuniq) * len(Yuniq))

    # --- Figure ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=300)
    if full_grid:
        Xsorted = np.sort(Xuniq)
        Ysorted = np.sort(Yuniq)
        H, W = len(Ysorted), len(Xsorted)
        grid = np.zeros((H, W), dtype=np.float32)
        x_pos = {x: i for i, x in enumerate(Xsorted)}
        y_pos = {y: i for i, y in enumerate(Ysorted)}
        for x, y, val in zip(xs, ys, v):
            grid[y_pos[int(y)], x_pos[int(x)]] = val
        im = ax.imshow(grid, origin="upper", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        sc = ax.scatter(xs, ys, c=v, s=4, cmap=cmap, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.invert_yaxis()
    ax.set_title(f"Latent map (full) — {tag}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    png_path = os.path.join(save_dir, f"latent_full_{tag}.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # --- CSV ---
    import pandas as pd
    df = pd.DataFrame({"x": xs, "y": ys, f"latent_{tag}": v})
    csv_path = os.path.join(save_dir, f"latent_full_{tag}.csv")
    df.to_csv(csv_path, index=False)

    return png_path, csv_path

# -----------------------------
# Latent vs reference (PCA/cNMF) via GMM
# -----------------------------
def pick_latent_by_clustering_gmm(
    latents, xs_all, ys_all,
    roi_coords, roi_feats,  # ROI-only features (PCA/cNMF)
    k_range=(2, 12), criterion="bic", metric="ari", topn=3, seed=0
):
    """
    1) Cluster ROI PCA/cNMF features with GMM (auto-k via BIC/AIC).
    2) For each latent dimension: cluster its ROI values with GMM (auto-k).
    3) Score similarity (ARI/NMI); return top-n latent dims.
    """
    # Align ROI to latent order
    src_idx, mask = align_indices_by_coords(xs_all, ys_all, roi_coords)
    if mask.sum() == 0:
        raise RuntimeError("No coordinate overlap between latent coords and ROI coords.")
    roi_feats = np.asarray(roi_feats, dtype=np.float32)
    lab_ref, k_ref, _ = gmm_auto_k(roi_feats[mask], k_min=k_range[0], k_max=k_range[1],
                                   criterion=criterion, random_state=seed)

    scores, details = [], []
    for d in range(latents.shape[1]):
        v = latents[src_idx[mask], d]
        lab_lat, k_lat, _ = gmm_auto_k_1d(v, k_min=k_range[0], k_max=k_range[1],
                                          criterion=criterion, random_state=seed)
        s = clustering_similarity(lab_lat, lab_ref, metric=metric)
        scores.append((d, float(s), int(k_lat), int(k_ref)))
        details.append((d, s, v.copy(), lab_lat.copy(), lab_ref.copy(), mask.copy(), src_idx.copy()))
    scores.sort(key=lambda t: t[1], reverse=True)
    topdims = [dim for dim, *_ in scores[:topn]]
    details = [d for d in details if d[0] in topdims]
    details.sort(key=lambda t: dict((x[0], x[1]) for x in scores)[t[0]], reverse=True)
    return scores[:topn], details

def plot_latent_vs_ref(save_dir, tag, xs_roi, ys_roi,
                       latent_vals_roi, labels_ref_roi, title_ref="Reference clusters (ROI)"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    _grid_or_scatter(axes[0], xs_roi, ys_roi, latent_vals_roi, "Latent map (ROI)", cmap="RdBu")
    sc = axes[1].scatter(xs_roi, ys_roi, c=labels_ref_roi, s=6, cmap="tab20", linewidths=0)
    axes[1].invert_yaxis()
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(title_ref)
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    out = os.path.join(save_dir, f"latent_vs_ref_{tag}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

# -----------------------------
# Synthetic element maps
# -----------------------------
def synthetic_element_map(
    elem_values_roi,              # (M,E) in ROI order
    mode="sum",
    weights=None,                 # None or length-E array (for sum)
    rgb_elements=None,            # list/tuple of 1-3 indices (for rgb)
    normalize_each=True
):
    """
    Build a synthetic element map from multiple element channels (ROI only).

    mode="sum":
      Optionally normalize each channel to [0,1] (per ROI) then weighted sum.
      If weights is None, equal weights.

    mode="rgb":
      Use up to 3 channels as (R,G,B). Return (M,3) float32 in [0,1] for saving as color image.

    Returns:
      map_1d : (M,) if mode="sum"
      or
      map_rgb: (M,3) if mode="rgb"
    """
    V = np.asarray(elem_values_roi, dtype=np.float32)
    if normalize_each:
        vmin = np.nanmin(V, axis=0, keepdims=True)
        vmax = np.nanmax(V, axis=0, keepdims=True)
        denom = np.maximum(vmax - vmin, 1e-8)
        V = (V - vmin) / denom

    if mode == "sum":
        if weights is None:
            weights = np.ones(V.shape[1], dtype=np.float32) / float(V.shape[1])
        else:
            weights = np.asarray(weights, dtype=np.float32)
            weights = weights / (np.sum(weights) + 1e-8)
        m = V @ weights
        return m.astype(np.float32)
    elif mode == "rgb":
        if rgb_elements is None or len(rgb_elements) == 0:
            raise ValueError("rgb_elements must be 1–3 channel indices for mode='rgb'.")
        rgb_e = list(rgb_elements)[:3]
        C = len(rgb_e)
        out = np.zeros((V.shape[0], 3), dtype=np.float32)
        for i, ch in enumerate(rgb_e):
            out[:, i] = V[:, ch]
        if C == 1:
            out[:, 1] = out[:, 0]  # gray
            out[:, 2] = out[:, 0]
        elif C == 2:
            out[:, 2] = 0.5 * (out[:, 0] + out[:, 1])
        return np.clip(out, 0.0, 1.0)
    else:
        raise ValueError("mode must be 'sum' or 'rgb'.")

def plot_latent_vs_element(save_dir, tag, xs_roi, ys_roi,
                           latent_vals_roi, elem_vals_roi, elem_name):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    _grid_or_scatter(axes[0], xs_roi, ys_roi, latent_vals_roi, "Latent map (ROI)", cmap="RdBu")
    axes[1].scatter(latent_vals_roi, elem_vals_roi, s=6, alpha=0.7)
    axes[1].set_xlabel("latent value"); axes[1].set_ylabel(elem_name)
    good = np.isfinite(latent_vals_roi) & np.isfinite(elem_vals_roi)
    if good.sum() >= 5:
        r, _ = pearsonr(latent_vals_roi[good], elem_vals_roi[good])
        axes[1].set_title(f"Latent vs {elem_name} (Pearson r={r:.3f})")
    else:
        axes[1].set_title(f"Latent vs {elem_name} (insufficient valid points)")
    fig.tight_layout()
    out = os.path.join(save_dir, f"latent_vs_{elem_name}_{tag}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

def save_rgb_map_png(save_dir, tag, xs_roi, ys_roi, rgb_flat):
    """
    Save an RGB element map (ROI) as PNG (dpi=300) with our orientation rule.
    """
    os.makedirs(save_dir, exist_ok=True)
    xs = np.asarray(xs_roi); ys = np.asarray(ys_roi)
    Xuniq = np.unique(xs); Yuniq = np.unique(ys)
    all_pairs = {(int(x), int(y)) for x in Xuniq for y in Yuniq}
    coords = {(int(x), int(y)) for x, y in zip(xs, ys)}
    full = (coords == all_pairs) and (len(coords) == len(Xuniq) * len(Yuniq))

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5), dpi=300)
    if full:
        Xsorted = np.sort(Xuniq)
        Ysorted = np.sort(Yuniq)
        H, W = len(Ysorted), len(Xsorted)
        grid = np.zeros((H, W, 3), dtype=np.float32)
        x_pos = {x: i for i, x in enumerate(Xsorted)}
        y_pos = {y: i for i, y in enumerate(Ysorted)}
        for (x, y), rgb in zip(zip(xs, ys), rgb_flat):
            grid[y_pos[int(y)], x_pos[int(x)]] = rgb
        ax.imshow(grid, origin="upper", aspect="auto")
    else:
        ax.scatter(xs, ys, c=rgb_flat, s=6)
        ax.invert_yaxis()
    ax.set_title("Synthetic element RGB (ROI)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    out = os.path.join(save_dir, f"element_rgb_{tag}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

# -----------------------------
# Orchestration
# -----------------------------
def run_latent_vs_pca_cnmf_gmm(
    out_dir,
    latents, xs_all, ys_all,
    roi_coords, roi_feats,
    kmin=2, kmax=12, criterion="bic", metric="ari", topn=3, seed=0
):
    """
    Select latent dims (top-n) whose ROI GMM clusters best match ROI PCA/cNMF GMM clusters.
    Saves per-latent ROI arrays and figures.
    """
    os.makedirs(out_dir, exist_ok=True)
    top, details = pick_latent_by_clustering_gmm(
        latents, xs_all, ys_all, roi_coords, roi_feats,
        k_range=(kmin, kmax), criterion=criterion, metric=metric, topn=topn, seed=seed
    )
    figs = []
    for (d, score, v_roi, lab_lat, lab_ref, mask, src_idx) in details:
        xs_roi = roi_coords[:, 0][mask].astype(int)
        ys_roi = roi_coords[:, 1][mask].astype(int)
        tag = f"d{d}_{metric}{score:.3f}"
        figs.append(
            plot_latent_vs_ref(out_dir, tag, xs_roi, ys_roi, v_roi, lab_ref, title_ref="ROI reference clusters")
        )
        np.save(os.path.join(out_dir, f"latent_d{d}_roi.npy"), v_roi.astype(np.float32))
        np.save(os.path.join(out_dir, f"latent_d{d}_full.npy"), latents[:, d].astype(np.float32))
        save_full_latent_map(out_dir, f"d{d}", xs_all, ys_all, latents[:, d], cmap="RdBu")
    report = {"top_latent_by_cluster": [{"dim": int(d), "score": float(s), "k_lat": int(kl), "k_ref": int(kr)}
                                        for (d, s, kl, kr) in top]}
    with open(os.path.join(out_dir, "report_cluster.json"), "w") as f:
        json.dump(report, f, indent=2)
    return figs, report

def run_latent_vs_elements(
    out_dir,
    latents, xs_all, ys_all,
    elem_coords_all, elem_values_all, elem_names,
    roi_xrange=None, roi_yrange=None,
    synth_mode="sum", weights=None, rgb_elements=None,
    topn=3
):
    """
    1) Select ROI via x/y ranges.
    2) Build synthetic element map (sum or RGB) in ROI.
    3) Rank latent dims by Pearson r w.r.t. synthetic map (ROI), save plots and arrays.
    """
    os.makedirs(out_dir, exist_ok=True)
    xs_all = np.asarray(xs_all, dtype=int)
    ys_all = np.asarray(ys_all, dtype=int)

    # Align to latent order
    src_idx, valid = align_indices_by_coords(xs_all, ys_all, elem_coords_all)
    xs_elem = elem_coords_all[:, 0].astype(int)
    ys_elem = elem_coords_all[:, 1].astype(int)

    # Select ROI
    roi_mask = roi_mask_from_ranges(xs_elem, ys_elem, roi_xrange, roi_yrange)
    roi_mask &= valid  # must exist in latent set

    if roi_mask.sum() == 0:
        raise RuntimeError("ROI selection returned zero valid samples.")

    # ROI values (align to latent indexing)
    idx = src_idx[roi_mask]
    elem_roi = elem_values_all[roi_mask]  # (M,E)
    xs_roi = xs_elem[roi_mask]
    ys_roi = ys_elem[roi_mask]

    # Build synthetic map
    if synth_mode == "rgb":
        synth = synthetic_element_map(elem_roi, mode="rgb", rgb_elements=rgb_elements, normalize_each=True)
        # For correlation ranking, reduce RGB to luminance to compare with 1D latent
        luminance = 0.2126 * synth[:, 0] + 0.7152 * synth[:, 1] + 0.0722 * synth[:, 2]
        scalar_map = luminance.astype(np.float32)
    else:
        synth = synthetic_element_map(elem_roi, mode="sum", weights=weights, normalize_each=True)
        scalar_map = synth

    # Rank latent dims by Pearson r in ROI
    best = []
    for d in range(latents.shape[1]):
        v_lat = latents[idx, d]
        ok = np.isfinite(v_lat) & np.isfinite(scalar_map)
        if ok.sum() >= 5:
            r, _ = pearsonr(v_lat[ok], scalar_map[ok])
            best.append((d, abs(float(r)), float(r)))
    best.sort(key=lambda t: t[1], reverse=True)

    # Save figures/arrays for topn
    figs = []
    for (d, ar, r) in best[:topn]:
        tag = f"d{d}_r{r:.3f}"
        figs.append(
            plot_latent_vs_element(out_dir, tag, xs_roi, ys_roi,
                                   latents[idx, d], scalar_map, elem_name="synthetic")
        )
        np.save(os.path.join(out_dir, f"latent_d{d}_roi_vs_synthetic.npy"),
                latents[idx, d].astype(np.float32))
        save_full_latent_map(out_dir, f"d{d}", xs_all, ys_all, latents[:, d], cmap="RdBu")
    # Optionally save RGB PNG for visualization
    if synth_mode == "rgb":
        figs.append(save_rgb_map_png(out_dir, "rgb", xs_roi, ys_roi, synth))

    report = {
        "element_names": list(elem_names),
        "roi": {"x_range": roi_xrange, "y_range": roi_yrange, "n": int(roi_mask.sum())},
        "top_latent_by_element": [{"dim": int(d), "abs_r": float(ar), "r": float(r)} for (d, ar, r) in best[:topn]]
    }
    with open(os.path.join(out_dir, "report_elements.json"), "w") as f:
        json.dump(report, f, indent=2)

    return figs, report
