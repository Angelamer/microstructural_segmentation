# -*- coding: utf-8 -*-
"""
Loop PCA component counts, select the best by agreement with downstream cNMF clustering,
then save PCA scores, cNMF weights, and coordinate labels to CSV.
Filenames include the chosen pc count, K for cNMF, and the ROI ranges.

Requirements: your project helpers must be importable:
- read_data, set_ROI, set_component
- get_components, coord_phase_dict_from_dataframe
- run_PCA, gmm_clustering, find_best_reference_window
- run_cNMF, evaluate_clustering_metrics
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import adjusted_rand_score

# ---- project imports  ----
from visualize_grid import read_data, set_ROI, set_component
from data_processing import get_components, coord_phase_dict_from_dataframe
from PCA import run_PCA
from cluster_analysis import (
    gmm_clustering,
    find_best_reference_window,
    evaluate_clustering_metrics,
    calculate_cluster_metrics,
    plot_cluster_distances_ranking
)
from cNMF import run_cNMF
from cNMF import set_global_determinism

set_global_determinism(seed=42, use_cuda=False)
# ---------------- Configuration ----------------
# Paths
PATH_PATTERNS = "/Volumes/T7/ebsd_data/Images_Valid/"
PATH_PHASEMAP = "/Volumes/T7/phase_map.png"
CSV_INDEXING = "/Volumes/T7/ebsd_processed_with_grain_boundary.csv"

# Kikuchi pattern & crop
HEIGHT = 512
WIDTH = 672
SLICE_X = (86, 586)
SLICE_Y = (56, 456)

# ROI you are working on
# ROI_X = (250, 300)
# ROI_Y = (30, 70)
ROI_X = (200,400)
ROI_Y = (0,100)
# PCA candidate counts to try (adjust as you like)
PCA_CANDIDATES = [100, 200, 400, 800]

# Number of components (K) for cNMF constraints (from 5 windows -> get_components(..., K=3, ...))
CNMF_K = 3

# GMM search settings (match your helpers)
GMM_INIT = None
GMM_MAX_CLUSTERS = 20  # upper bound to search inside gmm_clustering


# ----------------- Helper routines -----------------
def build_components_from_best_windows(best_window, path, grid, path_to_phase_map,
                                       height, width, slice_x, slice_y, pad=1):
    """Convert your best_window dict to R-list and components."""
    R_list = []
    ref_pos_list = []
    ranges_list = []  # (x_range, y_range, (cx, cy), key)

    for key in sorted(best_window.keys(), key=lambda k: int(k)):
        entry = best_window[key]
        cx, cy = map(int, entry["center_loc"])
        x_range = (max(0, cx - pad), min(width - 1, cx + pad))
        y_range = (max(0, cy - pad), min(height - 1, cy + pad))
        R, ref_pos = set_component(x_range, y_range, path, grid, path_to_phase_map)
        R_list.append(R)
        ref_pos_list.append(ref_pos)
        ranges_list.append((x_range, y_range, (cx, cy), int(key)))

    components = get_components(R_list, CNMF_K, None, height, width, slice_x, slice_y)
    return components, R_list, ref_pos_list, ranges_list


def score_pairing(coord_to_label_pca, coord_to_label_cnmf):
    """Score similarity between PCA and cNMF clusterings on shared coordinates."""
    coords = sorted(set(coord_to_label_pca.keys()) & set(coord_to_label_cnmf.keys()))
    if not coords:
        return 0.0, 0, 0, 0.0

    y1 = [coord_to_label_pca[c] for c in coords]
    y2 = [coord_to_label_cnmf[c] for c in coords]
    ari = adjusted_rand_score(y1, y2)

    n1 = len(set(y1))
    n2 = len(set(y2))
    same_k_bonus = 1.0 if n1 == n2 else 0.0  # strong preference for equal cluster counts

    # Final score: prioritize same K, then ARI
    score = same_k_bonus * 1.0 + (0.9 if same_k_bonus else 0.0) * ari + (0.5 if not same_k_bonus else 0.0) * ari
    # (equivalently: score = 1.0*I[n1==n2] + ari*(0.9 if equal else 0.5))

    return score, n1, n2, ari


def df_with_labels(loc, array2d, prefix, coord_to_label_lookup):
    """Build a dataframe for (x,y)+array columns, and attach cluster labels by lookup."""
    cols = [f"{prefix}_{i+1}" for i in range(array2d.shape[1])]
    df = pd.DataFrame(np.column_stack([loc, array2d]), columns=["x", "y"] + cols)
    df["cluster_label"] = [
        coord_to_label_lookup.get((int(x), int(y)), -1) for x, y in df[["x", "y"]].itertuples(index=False)
    ]
    return df


# ----------------------- Main -----------------------
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Read grid/maps
    grid = read_data(PATH_PATTERNS, PATH_PHASEMAP)

    # ROI block
    X_roi, loc_roi = set_ROI(ROI_X, ROI_Y, PATH, grid, PATH_PHASEMAP)  # noqa: E231

    # True phase map (for later evaluation/optional use)
    df_indexing = pd.read_csv(CSV_INDEXING)
    coord_phase_dict = coord_phase_dict_from_dataframe(df_indexing, fill_full_grid=True)

    # Try different PCA component counts
    trials = []
    best = None

    print(f"\n[INFO] Trying PCA counts: {PCA_CANDIDATES}\n")
    for n_pc in tqdm(PCA_CANDIDATES, desc="PCA n_components"):
        # ----- PCA -----
        pca_scores, pca_model = run_PCA(X_roi, n_pc, HEIGHT, WIDTH, SLICE_X, SLICE_Y)

        # ----- GMM on PCA scores -----
        gmm_model_pca, cluster_coords_pca, coord_to_label_pca, cluster_labels_pca, n_opt_pca, sil_pca = \
            gmm_clustering(pca_scores, loc_roi, GMM_INIT, GMM_MAX_CLUSTERS)

        # ----- pick best windows from PCA clusters -----
        centers, covs, variations = calculate_cluster_metrics(gmm_model_pca, cluster_labels_pca, pca_scores)
        top_samples_per_cluster= plot_cluster_distances_ranking(gmm_model_pca, cluster_labels_pca, pca_scores, loc_roi)
        best_window = find_best_reference_window(top_samples_per_cluster, cluster_labels_pca, variations, loc_roi)

        # ----- build components & run cNMF -----
        components, R_list, ref_pos_list, ranges_list = \
            build_components_from_best_windows(best_window, PATH_PATTERNS, grid, PATH_PHASEMAP,
                                               HEIGHT, WIDTH, SLICE_X, SLICE_Y, pad=1)

        weights, mse, r2 = run_cNMF(X_roi, components, HEIGHT, WIDTH, SLICE_X, SLICE_Y)

        # ----- GMM on cNMF weights -----
        gmm_model_cnmf, cluster_coords_cnmf, coord_to_label_cnmf, cluster_labels_cnmf, n_opt_cnmf, sil_cnmf = \
            gmm_clustering(weights, loc_roi, GMM_INIT, GMM_MAX_CLUSTERS)

        # ----- Score PCA↔cNMF agreement -----
        score, k_pca, k_cnmf, ari = score_pairing(coord_to_label_pca, coord_to_label_cnmf)

        trials.append({
            "n_pc": n_pc,
            "score": score,
            "ari": ari,
            "k_pca": k_pca,
            "k_cnmf": k_cnmf,
            "sil_pca": sil_pca,
            "sil_cnmf": sil_cnmf,
            "best_window": best_window,
            "pca_scores": pca_scores,
            "weights": weights,
            "coord_to_label_pca": coord_to_label_pca,
            "coord_to_label_cnmf": coord_to_label_cnmf,
            "ranges_list": ranges_list,
        })

        # Track the best so far
        if (best is None) or (score > best["score"]):
            best = trials[-1]

    # ---------------- Report best ----------------
    n_pc_best = best["n_pc"]
    print("\n" + "=" * 80)
    print(f"Best PCA components: {n_pc_best}")
    print(f"Score (K-match & ARI): {best['score']:.4f} | ARI={best['ari']:.4f} | "
          f"K_pca={best['k_pca']} K_cnmf={best['k_cnmf']} | "
          f"sil_pca={best['sil_pca']:.4f} sil_cnmf={best['sil_cnmf']:.4f}")
    print("-" * 80)
    print("Best windows (center_loc and +/-1 ranges):")
    for (x_range, y_range, center, key) in best["ranges_list"]:
        print(f"  key={key:>2d} center={center}  x_range={x_range}  y_range={y_range}")

    # ---------------- Save CSVs ----------------
    # filenames carry pc count, K and ROI ranges
    roi_tag = f"x{ROI_X[0]}-{ROI_X[1]}_y{ROI_Y[0]}-{ROI_Y[1]}"
    pca_name = f"pca_scores_pc{n_pc_best}_{roi_tag}.csv"
    cnmf_name = f"cnmf_weights_pc{n_pc_best}_{roi_tag}.csv"

    # PCA dataframe + labels (from PCA clustering)
    pca_df = df_with_labels(loc_roi, best["pca_scores"], "PCA",
                            coord_to_label_lookup=best["coord_to_label_pca"])
    pca_df.to_csv(pca_name, index=False)
    print(f"[SAVE] {pca_name}  (shape={pca_df.shape})")

    # cNMF dataframe + labels (from cNMF clustering)
    weights_df = df_with_labels(loc_roi, best["weights"], "cNMF",
                                coord_to_label_lookup=best["coord_to_label_cnmf"])
    weights_df.to_csv(cnmf_name, index=False)
    print(f"[SAVE] {cnmf_name}  (shape={weights_df.shape})")

    # Optional: also save best windows summary
    # win_rows = []
    # for (x_range, y_range, center, key) in best["ranges_list"]:
    #     win_rows.append({
    #         "key": key,
    #         "center_x": center[0], "center_y": center[1],
    #         "x0": x_range[0], "x1": x_range[1],
    #         "y0": y_range[0], "y1": y_range[1],
    #     })
    # win_df = pd.DataFrame(win_rows).sort_values("key")
    # win_csv = f"best_windows_pc{n_pc_best}_{roi_tag}.csv"
    # win_df.to_csv(win_csv, index=False)
    # print(f"[SAVE] {win_csv}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    # Unpack PATH_PATTERNS for set_ROI param naming used above
    PATH = PATH_PATTERNS
    main()
