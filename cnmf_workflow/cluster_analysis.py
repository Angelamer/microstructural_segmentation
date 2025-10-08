from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Patch
import matplotlib.cm as cm
from PCA import _add_confidence_ellipse
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, homogeneity_score, completeness_score, v_measure_score
import pandas as pd
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from utils import _safe_savefig
import json
import os
# from scipy.spatial.distance import mahalanobis


def _ranks(x, higher_is_better=False):
    """0 = best. Stable rank without scipy."""
    x = np.asarray(x, dtype=float)
    if higher_is_better:
        x = -x
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(x))
    return ranks

def gmm_clustering(
    scores,
    loc_roi,
    n_components=None,
    max_components=10,
    random_state=42,
    covariance_type="full",
    selection="hybrid",        # "hybrid" | "bic" | "aic" | "silhouette"
    weights=(0.6, 0.2, 0.2),   # (w_bic, w_aic, w_sil) for selection="hybrid"
    verbose=True,
):
    """
    GMM clustering with model selection via BIC/AIC/Silhouette.

    Args
    ----
    scores : (n_samples, n_features)
    loc_roi : (n_samples, 2) coordinates
    n_components : int or None
        If None, search k in [2, max_components] and select by `selection`.
    max_components : int
    random_state : int
    covariance_type : str
        GMM covariance_type ("full","tied","diag","spherical")
    selection : str
        "hybrid" : rank-combine BIC (↓), AIC (↓), Silhouette (↑)
        "bic"     : minimize BIC
        "aic"     : minimize AIC
        "silhouette": maximize silhouette
    weights : tuple
        Weights (w_bic, w_aic, w_sil) used when selection="hybrid".
    verbose : bool

    Returns
    -------
    gmm : GaussianMixture
    cluster_coords : dict[int -> list[(x,y)]]
    coord_to_label : dict[(x,y) -> int]
    cluster_labels : (n_samples,)
    optimal_n : int
    silhouette : float
    """
    scores = np.asarray(scores)
    loc_roi = np.asarray(loc_roi)
    assert len(scores) == len(loc_roi), "The sample number should be consistent."

    # Search k if not fixed
    if n_components is None:
        ks = np.arange(2, max_components + 1, dtype=int)
        models, labels_all = [], []
        bics, aics, sils = [], [], []

        for k in ks:
            gmm = GaussianMixture(
                n_components=k,
                random_state=random_state,
                covariance_type=covariance_type,
            ).fit(scores)
            labels = gmm.predict(scores)
            labels_all.append(labels)
            models.append(gmm)
            bics.append(gmm.bic(scores))
            aics.append(gmm.aic(scores))
            # silhouette only meaningful when k>1 and each cluster has ≥2 points
            try:
                sil = silhouette_score(scores, labels) if k > 1 else np.nan
            except Exception:
                sil = np.nan
            sils.append(sil)

        # Decide best k
        if selection == "bic":
            best_idx = int(np.argmin(bics))
        elif selection == "aic":
            best_idx = int(np.argmin(aics))
        elif selection == "silhouette":
            # if all NaN, fallback to BIC
            if np.all(np.isnan(sils)):
                best_idx = int(np.argmin(bics))
            else:
                sils_arr = np.nan_to_num(sils, nan=-1.0)  # treat NaN as very bad
                best_idx = int(np.argmax(sils_arr))
        else:  # "hybrid"
            w_bic, w_aic, w_sil = weights
            r_bic = _ranks(bics, higher_is_better=False)
            r_aic = _ranks(aics, higher_is_better=False)
            # For silhouette, NaN -> worst rank
            sils_arr = np.array(sils, dtype=float)
            sils_arr = np.nan_to_num(sils_arr, nan=-1.0)
            r_sil = _ranks(sils_arr, higher_is_better=True)
            combined = w_bic * r_bic + w_aic * r_aic + w_sil * r_sil
            best_idx = int(np.argmin(combined))

        optimal_n = int(ks[best_idx])
        gmm = models[best_idx]
        cluster_labels = labels_all[best_idx]
        silhouette = float(np.nan_to_num(sils[best_idx], nan=np.nan))

        if verbose:
            print(
                f"[GMM] selection={selection} -> k={optimal_n} | "
                f"BIC={bics[best_idx]:.1f} AIC={aics[best_idx]:.1f} "
                f"SIL={silhouette if not np.isnan(silhouette) else float('nan'):.4f}"
            )
    else:
        optimal_n = int(n_components)
        gmm = GaussianMixture(
            n_components=optimal_n,
            random_state=random_state,
            covariance_type=covariance_type,
        )
        cluster_labels = gmm.fit_predict(scores)
        try:
            silhouette = silhouette_score(scores, cluster_labels) if optimal_n > 1 else np.nan
        except Exception:
            silhouette = np.nan
        if verbose:
            print(f"[GMM] fixed k={optimal_n} | SIL={silhouette if not np.isnan(silhouette) else float('nan'):.4f}")

    # Build mappings
    cluster_coords = {}
    coord_to_label = {}
    for lab in np.unique(cluster_labels):
        mask = (cluster_labels == lab)
        cluster_coords[int(lab)] = [tuple(xy) for xy in loc_roi[mask]]
        for coord in loc_roi[mask]:
            coord_to_label[tuple(coord)] = int(lab)

    return gmm, cluster_coords, coord_to_label, cluster_labels, optimal_n, silhouette

def calculate_cluster_metrics(gmm_model, cluster_labels, scores):
    """
    Compute the statistical metrics of clustering
    
    Args:
        gmm_model (GaussianMixture): gmm model
        cluster_labels (np.array)
        scores (np.array)
    
    Returns:
        cluster_centers (dict):  {label: center}
        cluster_cov (dict):  {label: cov_matrix}
        variation (np.array): mahalanobis distance
    """
    # model parameters
    cluster_centers = gmm_model.means_
    cluster_cov = gmm_model.covariances_
    
    # Mahalanobis distance
    variation = np.zeros(len(scores))
    for i in range(len(scores)):
        label = cluster_labels[i]
        center = cluster_centers[label]
        cov = cluster_cov[label]
        
        try:
            if cov.ndim == 1:
                cov = np.diag(cov)
                
            inv_cov = np.linalg.inv(cov)
            diff = scores[i] - center
            variation[i] = np.sqrt(diff @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            # If Mahalanobis fails, Euclidean distance
            variation[i] = np.linalg.norm(scores[i] - center)
    
    # transform to the dic form
    centers_dict = {k: cluster_centers[k] for k in range(len(cluster_centers))}
    cov_dict = {k: cluster_cov[k] for k in range(len(cluster_cov))}
    
    return centers_dict, cov_dict, variation

def plot_cluster_distances_ranking(gmm_model, cluster_labels, scores, loc, top_n=10,
                                save_dir=None, filename=None, dpi=300, show=False):
    """
    Draw a line graph sorted by Mahalanobis distance within each cluster and return the top 10 samples with the smallest distance
    
    Args:
        gmm_model: model
        cluster_labels (np.array): cluster labels for each scan point
        scores (np.array)
        loc (np.array): coordinates of samples (n_samples, 2)
        top_n (int, optional): Defaults to 10.

    Returns:
        top_samples_per_cluster: dict {cluster_id: {'coordinates': coords, 'distances': dists, 'indices': indices}}
    """
    centers_dict, cov_dict, variation = calculate_cluster_metrics(gmm_model, cluster_labels, scores)

    # Obtain the unique labels
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    
    # Color setting
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    
    top_samples_per_cluster = {}

    for i, label in enumerate(unique_labels):
        # Get the samples within current cluster
        cluster_mask = cluster_labels == label
        cluster_indices = np.where(cluster_mask)[0]
        cluster_distances = variation[cluster_mask]
        cluster_coords = loc[cluster_mask]
        
        # Sort by the distance
        sorted_indices = np.argsort(cluster_distances)
        sorted_distances = cluster_distances[sorted_indices]
        sorted_coords = cluster_coords[sorted_indices]
        sorted_original_indices = cluster_indices[sorted_indices]
        
        # Line graph
        plt.plot(range(len(sorted_distances)), sorted_distances, 
                color=colors[i], marker='o', markersize=3, linewidth=2,
                label=f'Cluster {label} ({len(sorted_distances)} samples)')
        
        # Highlight the top_n points
        top_n_actual = min(top_n, len(sorted_distances))
        plt.scatter(range(top_n_actual), sorted_distances[:top_n_actual], 
                color=colors[i], s=50, edgecolor='black', linewidth=1, zorder=5)
        
        # Save the top_n sample information
        top_samples_per_cluster[label] = {
            'coordinates': sorted_coords[:top_n_actual],
            'distances': sorted_distances[:top_n_actual],
            'indices': sorted_original_indices[:top_n_actual]
        }
    
    ax.set_xlabel('Sample Rank (sorted by Mahalanobis distance)')
    ax.set_ylabel('Mahalanobis Distance')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    saved = _safe_savefig(fig, save_dir, filename, dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Print the information
    print("Top samples per cluster:")
    for label, info in top_samples_per_cluster.items():
        print(f"Cluster {label}: {len(info['coordinates'])} samples")
        print(f"  Distance range: {info['distances'][0]:.4f} - {info['distances'][-1]:.4f}")
        print(f"  Coordinates: {info['coordinates'][:3]}...") 
        print()
    
    return top_samples_per_cluster, saved

def find_best_reference_window(top_samples_per_cluster, cluster_labels, variation, loc, w1=1, w2=1):
    """
    Select the best reference window from the first 10 samples in each cluster
    
    Args:
        top_samples_per_cluster: dict {cluster_id: {'coordinates': coord, 'distances': value, 'indices': indices}}
        cluster_labels: (n_samples,)
        variation: Mahalanobis distance array (n_samples,)
        loc: coordinates of samples (n_samples, 2)
    
    Returns:
        best_window_info: dict
    """
    # Create a coordinate-to-index mapping dictionary
    coord_to_idx = {(x, y): idx for idx, (x, y) in enumerate(loc)}
    
    # Store metrics for all candidate windows
    candidate_windows = {cid: [] for cid in top_samples_per_cluster.keys()}
    
    # Process the top samples in each cluster
    for cid, samples_info in top_samples_per_cluster.items():
        sample_coords = samples_info['coordinates']
        sample_indices = samples_info['indices']
        
        for coord, orig_idx in zip(sample_coords, sample_indices):
            center_loc = tuple(coord)
            center_indice = orig_idx
            # obtain the 3X3 window
            window_indices = []
            center_x, center_y = center_loc
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_loc = (center_x + dx, center_y + dy)
                    
                    # Check the samples within each window in the roi
                    if neighbor_loc in coord_to_idx:
                        idx = coord_to_idx[neighbor_loc]
                        window_indices.append(idx)
            # the window has to be 3*3
            if len(window_indices) != 9:
                continue
            
            # Check whether all points in the window belong to the same cluster
            window_labels = cluster_labels[window_indices]
            if len(np.unique(window_labels)) > 1:
                continue
            
            # Calculate the Mahalanobis distance within the window
            window_distances = variation[window_indices]
            
            # Calculate the mean and variance within the window
            window_mean = np.mean(window_distances)
            window_variance = np.var(window_distances)
            
            # Calculate a new comprehensive index (the smaller the value, the better)
            metric = w1* window_mean + w2 * window_variance
            
            # Store the window information
            candidate_windows[cid].append({
                'cluster': cid,
                'center_loc': np.array(center_loc),
                'center_indice': center_indice,
                'window_points': window_indices,
                'window_mean': window_mean,
                'window_variance': window_variance,
                'metric': metric,
                'w1': w1,
                'w2': w2
            })
    
    best_windows = {}
    
    
    # If no reference candidate
    for cid, windows in candidate_windows.items():
        if not windows:
            print(f"Warning: No valid candidate windows found for cluster {cid}!")
            continue
    
    # Select the minimum metric
        best_window = min(windows, key=lambda x: x['metric'])
        best_windows[cid] = best_window
    
    
    # Print the window information
    print("="*50)
    print("Best Reference Windows per Cluster:")
    for cid, window_info in best_windows.items():
        print(f"Cluster {cid}:")
        print(f"  Center Location: {window_info['center_loc']}")
        print(f"  Metric Value: {window_info['metric']:.4f} (w1={w1}, w2={w2})")
        print(f"  Window Mean Distance: {window_info['window_mean']:.4f}")
        print(f"  Window Variance: {window_info['window_variance']:.4f}")
        print("-"*40)
    print("="*50)
    
    return best_windows

# plot the heatmap by the cluster labels
def plot_cluster_heatmap(cluster_coords, img_shape=(31, 31),
                        save_dir=None, filename=None, dpi=300, show=False):
    """
    Plot a cluster distribution heatmap, automatically mapping actual coordinates to the specified image dimensions.
    
    Args:
        cluster_coords (dict): Mapping of cluster labels to coordinate lists {label: [(x1,y1), (x2,y2), ...]}
        img_shape (tuple): Dimensions of the output heatmap (height, width)
    """
    # 1. Collect all coordinates and find min/max values
    all_coords = []
    for coords in cluster_coords.values():
        all_coords.extend(coords)
    
    if not all_coords:
        print("Warning: No coordinate data found")
        return
    
    all_coords = np.array(all_coords)
    min_x, min_y = np.min(all_coords, axis=0)
    max_x, max_y = np.max(all_coords, axis=0)
    
    heatmap = np.zeros(img_shape, dtype=int) -1
    
    # 3. Calculate scaling factors
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Avoid division by zero
    scale_x = (img_shape[1] - 1) / x_range if x_range > 0 else 1
    scale_y = (img_shape[0] - 1) / y_range if y_range > 0 else 1
    
    # 4. Create coordinate mapping function
    def map_coord(x, y):
        """Map actual coordinates to heatmap indices"""
        x_idx = int(round((x - min_x) * scale_x))
        y_idx = int(round((y - min_y) * scale_y))
        # Ensure within image bounds
        x_idx = max(0, min(x_idx, img_shape[1] - 1))
        y_idx = max(0, min(y_idx, img_shape[0] - 1))
        return y_idx, x_idx  # Note: Heatmap indices are (row, col) = (y, x)
    
    # 5. Populate heatmap with cluster labels
    for label, coords in cluster_coords.items():
        for (x, y) in coords:
            row, col = map_coord(x, y)
            heatmap[row, col] = label
            
    # color mapping
    unique_labels = sorted(cluster_coords.keys())
    n_colors = len(unique_labels)
    
    # Use tab10 colormap, adjust for -1 label (noise)
    base_colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_colors)))
    color_values = []
    for i, label in enumerate(unique_labels):
        if label == -1:
            color_values.append([0.7, 0.7, 0.7, 1])  # Gray for noise
        else:
            color_values.append(base_colors[i % 10])  # Cycle through tab10 colors
    
    cmap = ListedColormap(color_values)
    
    norm = plt.Normalize(vmin=min(unique_labels)-0.5, 
                        vmax=max(unique_labels)+0.5)
    
    
    # 7. Create legend elements
    legend_elements = [
        Patch(facecolor=color_values[i], 
            edgecolor='k', 
            label=f'Cluster {label}' if label != -1 else 'Noise')
        for i, label in enumerate(unique_labels)
    ]
    
    # 8. Plot heatmap without axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    ax.imshow(heatmap, cmap=cmap, interpolation='nearest')
    ax.axis('off')
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    saved = _safe_savefig(fig, save_dir, filename, dpi)
    if show: plt.show()
    else: plt.close(fig)
    return saved

def plot_intra_cluster_variation_map(loc_roi, variation, cluster_labels, img_shape=(31, 31),
                                    default_cluster_cmap_names=None, cluster_name_map=None, anomalies_cluster_pca_coords3=None, ref1_pos=None, ref2_pos=None, reference_windows=None,
                                    save_dir=None, filename=None, dpi=300, show=False):
    """
    Plot an intra-cluster variation heatmap for EBSD/EDS or clustering results.

    Each cluster is visualized with its own colormap, showing how samples vary 
    internally (e.g., Mahalanobis distance). Central points, reference points, 
    and anomalies are optionally highlighted.

    Args:
        loc_roi (array-like, shape (n_samples, 2)):
            Array of (x, y) coordinates for each sample (ROI positions).
        variation (array-like, shape (n_samples,)):
            Variation score for each sample (e.g., Mahalanobis distance, 
            intra-cluster deviation).
        cluster_labels (array-like, shape (n_samples,)):
            Cluster assignment (integer or categorical label) for each sample.
        img_shape (tuple, default=(31, 31)):
            Shape of the output image grid as (height, width).
        default_cluster_cmap_names (list of str, optional):
            List of matplotlib colormap names, one per cluster. If not provided, 
            a default set will be chosen.
        cluster_name_map (dict, optional):
            Mapping from cluster_label → human-readable name to show in colorbars.
        anomalies_cluster_pca_coords3 (array-like, optional):
            List/array of anomaly coordinates (x, y) to mark in black.
        ref1_pos (list of tuple, optional):
            List of (x, y) reference points to highlight with solid outlines 
            (colored according to cluster).
        ref2_pos (list of tuple, optional):
            List of (x, y) reference points to highlight with dashed outlines 
            (colored according to cluster).
        reference_windows (dict, optional):
            Dictionary of reference windows from e.g. `find_best_window`. 
            Expected format:
                {cluster_id: {"window_points": [indices, ...]}, ...}

    Returns:
        None. Displays a matplotlib figure with:
            - Heatmap of intra-cluster variation.
            - Highlighted "central" points (minimum variation per cluster).
            - Optional anomalies, reference windows, and reference points.
            - One colorbar per cluster showing normalized variation.
            - Legends for highlighted points.
    """
    
    loc_roi = np.asarray(loc_roi)
    
    if not (len(loc_roi) == len(cluster_labels) == len(variation)):
        raise ValueError("All input arrays must have the same length.")
    
    if len(loc_roi) == 0:
        print("No data to plot.")
        return
    # 1. Find min/max coordinates for mapping
    min_x, min_y = np.min(loc_roi, axis=0)
    max_x, max_y = np.max(loc_roi, axis=0)
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # 2. Calculate scaling factors
    scale_x = (img_shape[1] - 1) / x_range if x_range > 0 else 1
    scale_y = (img_shape[0] - 1) / y_range if y_range > 0 else 1
    
    # 3. Create mapping function
    def map_coord(x, y):
        """Map actual coordinates to heatmap indices"""
        x_idx = int(round((x - min_x) * scale_x))
        y_idx = int(round((y - min_y) * scale_y))
        x_idx = max(0, min(x_idx, img_shape[1] - 1))
        y_idx = max(0, min(y_idx, img_shape[0] - 1))
        return y_idx, x_idx
    
    # Get unique cluster labels and sort them
    unique_cluster_labels = np.unique(cluster_labels)
    num_clusters = len(unique_cluster_labels)
    
    # Set up default colormaps if not provided
    if default_cluster_cmap_names is None:
        default_cluster_cmap_names = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 
                                    'YlOrBr', 'BuGn', 'PuRd', 'Greys'][:num_clusters]

    
    # Create a large figure with space for colorbars
    fig = plt.figure(figsize=(15, 10))
    
    # Main heatmap axis
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # [left, bottom, width, height]
    
    # Create an empty RGBA image with transparency
    rgba_heatmap = np.zeros((img_shape[0], img_shape[1], 4))
    rgba_heatmap[:, :, 3] = 0.0  # Fully transparent
    
    # Dictionary to store cluster normalization info
    cluster_norms = {}
    cluster_cmaps = {}
    # Save the min variation points within clusters
    min_variation_points = []
    
    # Process each cluster
    for i, label_val in enumerate(unique_cluster_labels):
        mask = (cluster_labels == label_val)
        coords_in_cluster = loc_roi[mask]
        variations_in_cluster = variation[mask]
        
        # Skip if no points or all variations are NaN
        if coords_in_cluster.shape[0] == 0 or np.all(np.isnan(variations_in_cluster)):
            continue
        
        # Normalize variation scores for this specific cluster
        min_var = np.nanmin(variations_in_cluster)
        max_var = np.nanmax(variations_in_cluster)
        
        # Handle case where all variations in a cluster are the same
        if min_var == max_var:
            normalized_variations = np.full_like(variations_in_cluster, 0.5)
        else:
            normalized_variations = (variations_in_cluster - min_var) / (max_var - min_var)
        
        # Get colormap for this cluster
        cmap_name = default_cluster_cmap_names[i % len(default_cluster_cmap_names)]
        current_cmap = cm.get_cmap(cmap_name)
        cluster_cmaps[label_val] = current_cmap
        cluster_norms[label_val] = (min_var, max_var)
        
        # Get the most central point
        min_var_idx = np.nanargmin(variations_in_cluster)
        min_var_point = coords_in_cluster[min_var_idx]
        min_variation_points.append((min_var_point[0], min_var_point[1], label_val))
        # print("Cluster centers and corresponding labels:", min_variation_points)
        # Apply colors to the heatmap
        for idx, (x, y) in enumerate(coords_in_cluster):
            row, col = map_coord(x, y)
            norm_val = normalized_variations[idx]
            if not np.isnan(norm_val):
                color_rgba = current_cmap(norm_val)
                rgba_heatmap[row, col, :] = color_rgba
    print("Cluster centers and corresponding labels:", min_variation_points)
    # Display the RGBA heatmap
    ax.imshow(rgba_heatmap, interpolation='nearest', origin='upper')
    # ax.set_title("Intra-Cluster Variation (Mahalanobis Distance)", fontsize=16)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    
    # Box select the point with the min distance
    min_var_patches = []
    for x, y, label_val in min_variation_points:
        row, col = map_coord(x, y)
        rect = Rectangle((col - 0.5, row - 0.5), 1, 1, 
                        linewidth=3, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        min_var_patches.append(rect)
    # Reference points box
    ref_win_patches = []
    if reference_windows is not None:
        for cid, win_info in reference_windows.items():
            window_points = win_info['window_points']
            for idx in window_points:
                x, y = loc_roi[idx]
                row, col = map_coord(x, y)
                rect = Rectangle((col - 0.5, row - 0.5), 1, 1, 
                                linewidth=1.5, edgecolor='red', 
                                facecolor='none', hatch='////', alpha=0.8)
                ax.add_patch(rect)
                ref_win_patches.append(rect)
    ref1_patches = []
    ref2_patches = []
    # Plot ref1_pos with cluster colors
    if ref1_pos is not None:
        for x, y in ref1_pos:
            row, col = map_coord(x, y)
            # Find cluster label for this point
            distances = np.linalg.norm(loc_roi - [x, y], axis=1)
            nearest_idx = np.argmin(distances)
            cluster_label = cluster_labels[nearest_idx]
                    
            if cluster_label in cluster_cmaps:
                color = cluster_cmaps[cluster_label](0.5)  # Mid-point color
                rect = Rectangle((col - 0.5, row - 0.5), 1, 1, 
                                linewidth=2, edgecolor=color, 
                                facecolor='none', linestyle='-')
                ax.add_patch(rect)
                ref1_patches.append(rect)
        
    # Plot ref2_pos with cluster colors
    if ref2_pos is not None:
        for x, y in ref2_pos:
            row, col = map_coord(x, y)
            distances = np.linalg.norm(loc_roi - [x, y], axis=1)
            nearest_idx = np.argmin(distances)
            cluster_label = cluster_labels[nearest_idx]
                
            if cluster_label in cluster_cmaps:
                color = cluster_cmaps[cluster_label](0.5)  # Mid-point color
                rect = Rectangle((col - 0.5, row - 0.5), 1, 1, 
                                linewidth=2, edgecolor=color, 
                                facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ref2_patches.append(rect)
    anomaly_patch_handles = []
    if anomalies_cluster_pca_coords3 is not None:
        anomalies_cluster_pca_coords3 = np.asarray(anomalies_cluster_pca_coords3)
        for x, y in anomalies_cluster_pca_coords3:
            row, col = map_coord(x, y)
            rect = Rectangle((col - 0.5, row - 0.5), 1, 1,
                            linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            anomaly_patch_handles.append(rect)
    # Add "center point" legend
    legend_handles = []
    if min_var_patches:
        center_legend = Patch(facecolor='none', edgecolor='yellow', linewidth=3, label='"Central" Point')
        legend_handles.append(center_legend)
    if ref_win_patches:
        ref_win_legend = Patch(facecolor='none', edgecolor='red', hatch='////', linewidth=1.5, 
                            label='Reference Window Points', alpha=0.8)
        legend_handles.append(ref_win_legend)
    if ref1_patches:
        ref1_legend = Patch(facecolor='none', edgecolor='gray', linewidth=2, 
                            linestyle='-', label='Ref1 Position')
        legend_handles.append(ref1_legend)
    if ref2_patches:
        ref2_legend = Patch(facecolor='none', edgecolor='gray', linewidth=2, 
                            linestyle='--', label='Ref2 Position')
        legend_handles.append(ref2_legend)
    if anomaly_patch_handles:
        anomaly_legend = Patch(facecolor='none', edgecolor='black', linewidth=2, label='Anomaly Point')
        legend_handles.append(anomaly_legend)

    if legend_handles:
        fig.legend(handles=legend_handles,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(legend_handles),
                frameon=False)
    plt.subplots_adjust(bottom=0.15)
    cax_width = 0.02
    cax_height = 0.8
    cax_spacing = 0.03
    cax_top = 0.9
    # Create colorbars for each cluster
    # cbar_axes = []
    cax_left = 0.7
    for i, label_val in enumerate(unique_cluster_labels):
        if label_val not in cluster_norms:
            continue
        
        # Calculate position for this colorbar
        cax_bottom = cax_top-cax_height
        
        # Create colorbar axis
        cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
        
        # Get colormap and normalization
        cmap = cluster_cmaps[label_val]
        min_val, max_val = cluster_norms[label_val]
        norm = Normalize(vmin=min_val, vmax=max_val)
        
        # Create ScalarMappable for colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Create colorbar
        cbar = Colorbar(cax, sm, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        
        # Set cluster name
        cluster_name = f"Cluster {label_val}"
        if cluster_name_map and label_val in cluster_name_map:
            cluster_name = cluster_name_map[label_val]
        
        cbar.set_label(cluster_name, ha='center', va='bottom', labelpad=15)
        cax_left += cax_width + cax_spacing
    
    plt.tight_layout()
    saved = _safe_savefig(fig, save_dir, filename, dpi)
    if show: plt.show()
    else: plt.close(fig)
    return saved

    

    
def dbscan_clustering(scores, loc_roi, eps=None, min_samples=5, eps_range=None):
    """
    DBSCAN clustering
    
    Args:
        scores (np.array): reduced data
        loc_roi (array-like): coordinates list
        eps (float/None): neighborhood radius，define the max distance between samples
        min_samples (int): minimum neighbor samples
        eps_range (list): 
    
    Returns:
        cluster_coords (dict): dicts of coordinates; noise=-1
        cluster_labels (np.array)
        optimal_eps (float): 
        silhouette (float)
    """
    scores = np.asarray(scores)
    loc_roi = np.asarray(loc_roi)
    assert len(scores) == len(loc_roi), "The sample number should be consistent."
    
    # auto-search the optimal eps
    if eps is None:
        if eps_range is None:
            eps_range = np.linspace(0.1, 1.0, 10)  # default
        
        best_eps = None
        best_sil = -1
        best_labels = None
        
        for eps_candidate in eps_range:
            db = DBSCAN(eps=eps_candidate, min_samples=min_samples)
            labels = db.fit_predict(scores)
            
            # calculate the silhouette scores for the valid clusters
            mask = labels != -1
            if len(np.unique(labels[mask])) < 2:  
                continue
            
            try:
                sil = silhouette_score(scores[mask], labels[mask])
                if sil > best_sil:
                    best_sil = sil
                    best_eps = eps_candidate
                    best_labels = labels
            except:
                continue
        
        if best_eps is None:
            raise ValueError("Can not find the valid eps")
        
        cluster_labels = best_labels
        silhouette = best_sil
        optimal_eps = best_eps
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = db.fit_predict(scores)
        optimal_eps = eps
        
        mask = cluster_labels != -1
        if len(np.unique(cluster_labels[mask])) >= 2:
            silhouette = silhouette_score(scores[mask], cluster_labels[mask])
        else:
            silhouette = np.nan
    
    # Categorized based on the labels
    cluster_coords = {}
    for label in np.unique(cluster_labels):
        mask = (cluster_labels == label)
        cluster_coords[int(label)] = loc_roi[mask].tolist() 
    
    # Cluster number(exclude the noise)
    n_clusters = len(np.unique(cluster_labels)) - (-1 in cluster_labels)
    
    return cluster_coords, cluster_labels, optimal_eps, n_clusters, silhouette

def plot_cnmf_scatter_with_boundary(weights, loc, cluster_labels, optimal_n, 
                                ref1_pos, ref2_pos, anomalies_dict=None, d=0.1, title_prefix="GMM Clustering for cNMF weights", 
                                ellipse_alpha=0.3):
    """
    Plot cnmf clustering results with boundary analysis in 2D space.

    Args:
        weights (ndarray): transformed data, shape (n_samples, n_components)
        loc (ndarray): coordinates of samples, shape (n_samples, 2)
        cluster_labels (ndarray): Cluster labels
        optimal_n (int): Optimal number of clusters
        variations (ndarray): Mahalanobis distance of each sample within one cluster
        ref1_pos (ndarray): Coordinates for reference position 1
        ref2_pos (ndarray): Coordinates for reference position 2
        d (float): Distance for boundary definition
        title_prefix (str): Plot title prefix
        ellipse_alpha (float): Transparency for confidence ellipses
        
    Returns:
        boundary_mask (ndarray): Boolean mask for boundary points
        boundary_scores (ndarray): Scores of boundary points
        boundary_locs (ndarray): Locations of boundary points
        slope (float): Slope of linear regression line
        intercept (float): Intercept of linear regression line
    """
    assert weights.shape[1] >= 2, "scores must have at least 2 components."
    assert loc.shape[0] == weights.shape[0], "loc and scores must have same number of samples."
    
    # Find indices for reference positions
    ref1_idx = [np.where((loc == pos).all(axis=1))[0][0] for pos in ref1_pos]
    ref2_idx = [np.where((loc == pos).all(axis=1))[0][0] for pos in ref2_pos]
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 8))
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels)
    
    # Use distinct colors for clusters
    default_cluster_cmap_names = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 
                                'YlOrBr', 'BuGn', 'PuRd', 'Greys'][:num_clusters]
    colors = [cm.get_cmap(name)(0.6) for name in default_cluster_cmap_names]
    
    # Create color map for anomalies based on cluster colors
    anomaly_colors = {}
    for label in unique_labels:
        idx = np.where(unique_labels == label)[0][0]
        anomaly_colors[label] = colors[idx]
    # Boundary points
    distances = np.abs(weights[:, 0] - weights[:, 1]) / np.sqrt(2)
    boundary_mask = distances < d
    boundary_weights = weights[boundary_mask]
    boundary_labels = cluster_labels[boundary_mask]  # NEW: Extract boundary labels
    boundary_added = False
    for i, label in enumerate(unique_labels):
        mask = (cluster_labels == label) & (~boundary_mask)
        ax.scatter(weights[mask, 0], weights[mask, 1], 
                color=colors[i], alpha=0.7, label=f'Cluster {label}', edgecolors='k')
        _add_confidence_ellipse(ax, weights[cluster_labels == label, :2], colors[i], alpha=ellipse_alpha)
    boundary_added = False
    for i, label in enumerate(unique_labels):
        mask = (cluster_labels == label) & (boundary_mask)
        if np.any(mask):
            label_text = 'Boundary Points' if not boundary_added else None
            ax.scatter(weights[mask, 0], weights[mask, 1], 
                    color=colors[i], alpha=0.9, marker='+', s=100, 
                    label=label_text)
            boundary_added = True
    ax.scatter(weights[ref1_idx, 0], weights[ref1_idx, 1],
            s=200, marker='*', facecolor='yellow', edgecolor='k', 
            linewidth=1.5, zorder=10, label='Ref1 Positions')
    
    ax.scatter(weights[ref2_idx, 0], weights[ref2_idx, 1],
            s=200, marker='*', facecolor='yellow', edgecolor='k', 
            linewidth=1.5, zorder=10, label='Ref2 Positions')
    
    # Plot anomalies if provided
    if anomalies_dict is not None:
        # Find indices for anomaly coordinates
        anomaly_indices = []
        anomaly_labels = []
        
        for coord, label in anomalies_dict.items():
            # Find matching location in loc array
            idx = np.where((loc == coord).all(axis=1))
            if idx[0].size > 0:
                anomaly_indices.append(idx[0][0])
                anomaly_labels.append(label)
        
        if anomaly_indices:
            # Convert to numpy arrays
            anomaly_indices = np.array(anomaly_indices)
            anomaly_labels = np.array(anomaly_labels)
            
            # Plot each anomaly with triangle marker and corresponding color
            for label in np.unique(anomaly_labels):
                mask = (anomaly_labels == label)
                ax.scatter(weights[anomaly_indices[mask], 0], 
                        weights[anomaly_indices[mask], 1],
                        marker='^', s=60, alpha=0.8, 
                        facecolor=anomaly_colors.get(label, 'gray'),
                        edgecolor='black', linewidth=1.0,
                        zorder=9,  # Below reference points
                        label='Anomalies' if label == anomaly_labels[0] else None)
    
    ax.set_xlim(0, np.max(weights[:, 0]))
    ax.set_ylim(0, np.max(weights[:, 1]))
    # Plot X+Y-1=0 line
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Line for X+Y-1=0
    x_line = np.linspace(min(x_min, y_min-1), max(x_max, y_max-1), 100)
    y_line = 1 - x_line
    ax.plot(x_line, y_line, 'r-', linewidth=2, label='X+Y-1=0')
    
    # Line for X=Y
    diag_line = np.linspace(min(x_min, y_min), max(x_max, y_max), 100)
    ax.plot(diag_line, diag_line, 'g-', linewidth=2, label='X=Y')
    
    # Create boundary lines and region
    x_vals = np.linspace(x_min, x_max, 100)
    
    # Upper boundary: y = x + sqrt(2)*d
    upper_bound = x_vals + np.sqrt(2) * d
    
    # Lower boundary: y = x - sqrt(2)*d
    lower_bound = x_vals - np.sqrt(2) * d
    
    # Plot the boundary
    ax.plot(x_vals, upper_bound, 'g--', linewidth=1.5, label=f'Boundary ±{d}')
    ax.plot(x_vals, lower_bound, 'g--', linewidth=1.5)
    
    # Fill the boundary
    ax.fill_between(x_vals, upper_bound, lower_bound, 
                    where=(upper_bound >= lower_bound), 
                    color='green', alpha=0.2, interpolate=True)
    
    # Linear regression
    X = weights[:, 0].reshape(-1, 1)
    y = weights[:, 1]
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    # Plot regression
    reg_x = np.linspace(x_min, x_max, 100)
    reg_y = slope * reg_x + intercept
    ax.plot(reg_x, reg_y, 'b-', linewidth=2, label='Linear Regression')
    
    # Obtain the boundary points
    boundary_scores = weights[boundary_mask]
    boundary_locs = loc[boundary_mask]
    
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_title(f'{title_prefix} (n_clusters={optimal_n})')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return boundary_mask, boundary_scores, boundary_locs, boundary_labels, slope, intercept

def plot_cnmf_weights_projected(
    weights,                 # (N, K)
    loc,                     # (N, 2) coordinates (x,y)
    cluster_labels=None,     # (N,) or None
    comps=(0, 1),            # tuple of component indices to plot (len=2 or 3)
    mode="2d",               # "2d" or "3d"
    # --- boundary points are now provided explicitly ---
    boundary_locs=None,      # list/array of (x,y); these will be marked as boundary points
    ref_pos_list=None,       # list/dict of reference coords; see docstring
    anomalies_dict=None,     # {(x,y): cluster_label_or_tag, ...}
    title="cNMF weights (projected)",
    ellipse_alpha=0.25,
    xlim=None, ylim=None, zlim=None, normalize=True
):
    """
    Plot cNMF weights in a chosen 2D/3D component subspace, show:
      - boundary points: near-equal contributions among the selected components
      - the 'simplex' constraint (sum of chosen comps = 1) line/plane
      - cluster coloring (optional)
      - reference positions (by loc), one set per component group if provided
      - anomalies (by loc)

    Generalized boundary definition:
        For the selected m components (m=2 or 3) with values w_i, define:
            mean = (w_1 + ... + w_m) / m
            boundary_metric = ||w - mean||_2 / sqrt(m)
        This equals |w1 - w2| / sqrt(2) when m=2, and measures distance to the
        equal-contribution subspace (w1 = w2 = ... = w_m).
        Points with boundary_metric < equal_band_d are considered boundary points.

    Args:
        weights (np.ndarray): (N, K) cNMF weights (usually nonnegative, often sum to 1).
        loc (np.ndarray): (N, 2) sample coordinates (used to find indices for ref/anomaly).
        cluster_labels (np.ndarray or None): (N,), optional cluster id per sample.
        comps (tuple[int]): component indices to project. Length must be 2 (2D) or 3 (3D).
        mode (str): "2d" or "3d" projection.
        equal_band_d (float): thickness threshold for the boundary band in the selected subspace.
        ref_pos_list (list[list[tuple]] or None):
            Example for K components: [ref_for_comp0, ref_for_comp1, ..., ref_for_compK-1],
            where each ref_for_compX is a list of (x,y) tuples.
            Only the references that fall into the selected `comps` are drawn, with distinct markers.
        anomalies_dict (dict or None): {(x,y): label} to mark special samples.
        title (str): plot title.
        ellipse_alpha (float): transparency for 2D confidence ellipses per cluster.
        xlim, ylim, zlim: axis limits (optional).

    Returns:
        out (dict):
            {
              "mask_boundary": (N,) bool array marking boundary points in selected subspace,
              "proj": (N,m) projected weights,
              "boundary_points": projected coords of boundary points,
              "boundary_indices": indices of boundary points,
              "regression_2d": {"slope": float, "intercept": float} or None,
            }
    """
    weights = np.asarray(weights, dtype=float)
    loc = np.asarray(loc)
    N, K = weights.shape
    comps = tuple(comps)
    assert len(comps) in (2, 3), "comps must have length 2 (2D) or 3 (3D)."
    assert all(0 <= c < K for c in comps), "comps indices out of range."
    assert loc.shape[0] == N, "loc and weights must have same number of rows."

    m = len(comps)

    # ---- Normalize per row so sum=1 (robust to zero rows) ----
    if normalize:
        sums = weights.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        Wnorm = weights / sums  # (N, K)
        row_sums = Wnorm.sum(axis=1)
        print("Row sums (min, max, mean):", row_sums.min(), row_sums.max(), row_sums.mean())
    else:
        Wnorm = weights
        
    Wplot = Wnorm[:, comps]  # use normalized weights for display
    # ----- boundary mask from boundary_locs (explicit input) -----
    def _coords_to_indices(coords_2d):
        """Map a list/array of (x,y) coords to row indices in `loc`."""
        if coords_2d is None:
            return np.array([], dtype=int)
        arr = np.asarray(coords_2d)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            return np.array([], dtype=int)
        idxs = []
        for pos in arr:
            ix = np.where((loc == pos).all(axis=1))[0]
            if ix.size > 0:
                idxs.append(ix[0])
        return np.asarray(idxs, dtype=int) if idxs else np.array([], dtype=int)

    boundary_idx = _coords_to_indices(boundary_locs)
    mask_boundary = np.zeros(N, dtype=bool)
    if boundary_idx.size > 0:
        mask_boundary[boundary_idx] = True

    # ----- helpers for references/anomalies -----
    def _get_ref_coords_for_comp(comp_idx):
        if ref_pos_list is None:
            return None
        if isinstance(ref_pos_list, dict):
            coords = ref_pos_list.get(comp_idx, None)
        else:
            coords = ref_pos_list[comp_idx] if (isinstance(ref_pos_list, (list, tuple)) and comp_idx < len(ref_pos_list)) else None
        if coords is None:
            return None
        arr = np.asarray(coords)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            return None
        # keep dtype consistent with loc for exact equality
        if np.issubdtype(loc.dtype, np.integer):
            arr = arr.astype(int, copy=False)
        return arr

    def _coords_to_indices_any(coords_2d):
        return _coords_to_indices(coords_2d)

    # ----- cluster colors -----
    if cluster_labels is None:
        cluster_labels = np.zeros(N, dtype=int)
    cluster_labels = np.asarray(cluster_labels)
    uniq = np.unique(cluster_labels)
    num_clusters = len(uniq)
    palette = [cm.get_cmap(name)(0.65) for name in
               ['Blues','Greens','Reds','Purples','Oranges','YlOrBr',
                'BuGn','PuRd','Greys','PuBu','YlGn','GnBu']][:num_clusters]
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}

    # ----- plotting -----
    if mode.lower() == "2d":
        fig, ax = plt.subplots(figsize=(10, 7))

        # main scatter by cluster
        for lab in uniq:
            # non-boundary first
            msk = (cluster_labels == lab) & (~mask_boundary)
            if np.any(msk):
                ax.scatter(Wplot[msk, 0], Wplot[msk, 1],
                           s=30, alpha=0.75, edgecolors='k', linewidths=0.5,
                           c=[color_map[lab]], label=f"Cluster {lab}")
            # optional ellipse per cluster
            _add_confidence_ellipse(ax, Wplot[cluster_labels == lab, :2], color_map[lab], alpha=ellipse_alpha)

        # boundary points (from input)
        if np.any(mask_boundary):
            ax.scatter(Wplot[mask_boundary, 0], Wplot[mask_boundary, 1],
                       s=35, marker="x", c="k", linewidths=1.1, alpha=0.9,
                       label="Boundary (given)")

        # simplex edge X + Y = 1
        xmin = 0.0 if xlim is None else xlim[0]
        xmax = 1.0 if xlim is None else xlim[1]
        x_line = np.linspace(xmin, xmax, 400)
        y_simplex = 1.0 - x_line
        ax.plot(x_line, y_simplex, 'r-', lw=1.6, alpha=0.9, label="X+Y=1")

        # regression (Y ~ X) on all projected points
        reg = LinearRegression().fit(Wplot[:, [0]], Wplot[:, 1])
        y_reg = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_reg, 'b--', lw=1.4, alpha=0.8, label="Linear fit")

        # references for selected comps
        ref_markers = ['*', 'P', 'X', 'D', '^', 's']
        for j, comp_idx in enumerate(comps):
            coords = _get_ref_coords_for_comp(comp_idx)
            if coords is None:
                continue
            idxs = _coords_to_indices_any(coords)
            if idxs.size == 0:
                continue
            ax.scatter(Wplot[idxs, 0], Wplot[idxs, 1],
                       s=200, marker=ref_markers[j % len(ref_markers)],
                       facecolor='yellow', edgecolor='k', linewidth=1.2,
                       label=f"Ref (comp {comp_idx})", zorder=10)

        # anomalies
        if anomalies_dict:
            an_xy = list(anomalies_dict.items())  # [((x,y), lbl), ...]
            a_idx, a_lab = [], []
            for (xy, lab) in an_xy:
                ix = np.where((loc == xy).all(axis=1))[0]
                if ix.size > 0:
                    a_idx.append(ix[0]); a_lab.append(lab)
            if a_idx:
                a_idx = np.asarray(a_idx, int); a_lab = np.asarray(a_lab, object)
                for lab in np.unique(a_lab):
                    msk = (a_lab == lab)
                    ax.scatter(Wplot[a_idx[msk], 0], Wplot[a_idx[msk], 1],
                               marker='^', s=70, facecolor='none', edgecolor='black',
                               linewidth=1.2, label=f"Anomaly: {lab}", zorder=9)

        ax.set_xlabel(f"Weight[{comps[0]}]")
        ax.set_ylabel(f"Weight[{comps[1]}]")
        ax.set_title(title)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        reg_info = {"slope": float(reg.coef_[0]), "intercept": float(reg.intercept_)}

    elif mode.lower() == "3d":
        assert len(comps) == 3, "For 3D mode, comps must have length 3."
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for lab in uniq:
            msk = (cluster_labels == lab) & (~mask_boundary)
            if np.any(msk):
                ax.scatter(Wplot[msk, 0], Wplot[msk, 1], Wplot[msk, 2],
                           s=20, alpha=0.8, c=[color_map[lab]], label=f"Cluster {lab}")

        if np.any(mask_boundary):
            ax.scatter(Wplot[mask_boundary, 0], Wplot[mask_boundary, 1], Wplot[mask_boundary, 2],
                       s=30, marker="x", c="k", alpha=0.9, label="Boundary (given)")

        # simplex plane w_i + w_j + w_k = 1
        g = np.linspace(0, 1, 30)
        X, Y = np.meshgrid(g, g)
        Z = 1.0 - X - Y
        Zmask = Z >= 0
        ax.plot_surface(np.where(Zmask, X, np.nan),
                        np.where(Zmask, Y, np.nan),
                        np.where(Zmask, Z, np.nan),
                        color='r', alpha=0.15, linewidth=0, antialiased=False)

        # references
        ref_markers = ['*', 'P', 'X', 'D', '^', 's']
        for j, comp_idx in enumerate(comps):
            coords = _get_ref_coords_for_comp(comp_idx)
            if coords is None:
                continue
            idxs = _coords_to_indices_any(coords)
            if idxs.size == 0:
                continue
            ax.scatter(Wplot[idxs, 0], Wplot[idxs, 1], Wplot[idxs, 2],
                       s=120, marker=ref_markers[j % len(ref_markers)],
                       facecolor='yellow', edgecolor='k', linewidth=1.1,
                       label=f"Ref (comp {comp_idx})", zorder=10)

        # anomalies
        if anomalies_dict:
            an_xy = list(anomalies_dict.items())
            a_idx, a_lab = [], []
            for (xy, lab) in an_xy:
                ix = np.where((loc == xy).all(axis=1))[0]
                if ix.size > 0:
                    a_idx.append(ix[0]); a_lab.append(lab)
            if a_idx:
                a_idx = np.asarray(a_idx, int); a_lab = np.asarray(a_lab, object)
                for lab in np.unique(a_lab):
                    msk = (a_lab == lab)
                    ax.scatter(Wplot[a_idx[msk], 0], Wplot[a_idx[msk], 1], Wplot[a_idx[msk], 2],
                               marker='^', s=60, facecolor='none', edgecolor='black',
                               linewidth=1.2, label=f"Anomaly: {lab}", zorder=9)

        ax.set_xlabel(f"Weight[{comps[0]}]")
        ax.set_ylabel(f"Weight[{comps[1]}]")
        ax.set_zlabel(f"Weight[{comps[2]}]")
        ax.set_title(title)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        if zlim: ax.set_zlim(*zlim)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        reg_info = None
    else:
        raise ValueError("mode must be '2d' or '3d'.")

    out = {
        "mask_boundary": mask_boundary,
        "proj": Wplot,
        "boundary_points": Wplot[mask_boundary],
        "boundary_indices": np.where(mask_boundary)[0],
        "boundary_dict": {
            tuple(loc[i]): int(cluster_labels[i]) if cluster_labels is not None else None
            for i in np.where(mask_boundary)[0]
        },
        "regression_2d": reg_info if mode.lower() == "2d" else None,
    }
    return out



def plot_gmm_clusters(scores, cluster_labels, variations, dim=2, anomalies=None, reference_windows=None, ellipse_alpha=0.3,
                    save_dir=None, filename=None, dpi=300, show=False):
    """
    Plot GMM clustering results in 2D or 3D based on selected PCA components.

    Args:
        scores (ndarray): transformed data, shape (n_samples, n_components)
        cluster_labels (ndarray): Cluster labels
        optimal_n (int): Optimal number of clusters
        variations (ndarray): Mahalanobis distance of each sample within one cluster
        dim (int): 2 or 3, to plot in 2D or 3D space using first dim principal components
        anomalies (ndarray or None): Optional anomalies to be marked, shape (n_anomalies, 2)
        reference_windows (dict or None): Optional reference windows from find_best_window
        title_prefix (str): Plot title prefix
        ellipse_alpha (float): Transparency for confidence ellipses
    """
    assert dim in [2, 3], "Only dim=2 or dim=3 is supported."
    assert scores.shape[1] >= dim, f"pca_scores must have at least {dim} components."

    # Set axis labels
    axis_labels = ['Component 1', 'Component 2', 'Component 3']
    
    unique_labels = np.unique(cluster_labels)
    center_points = []
    center_labels = []
    for label in unique_labels:
        mask = (cluster_labels == label)
        if np.sum(mask) == 0:
            continue
        # Find the minimum Mahalanobis distance
        min_var_idx = np.argmin(variations[mask])
        # obtain the index for center points
        global_idx = np.where(mask)[0][min_var_idx]
        center_points.append(scores[global_idx])
        center_labels.append(label)
    center_points = np.array(center_points)
    num_clusters = len(unique_labels)
    default_cluster_cmap_names = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 
                                'YlOrBr', 'BuGn', 'PuRd', 'Greys',
                                'YlGnBu','YlOrRd', 'OrRd', 'RdPu', 'GnBu', 
                                'PuBuGn','BuPu', 'RdGy', 'YlGn', 'PuBu', 'Oranges_r'][:num_clusters]
    colors = [cm.get_cmap(name)(0.6) for name in default_cluster_cmap_names]
    ref_colors = [cm.get_cmap(name)(0.9) for name in default_cluster_cmap_names]
    if dim == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            mask = (cluster_labels == label)
            ax.scatter(scores[mask, 0], 
                    scores[mask, 1], 
                    scores[mask, 2],
                    color=colors[i], alpha=0.7, label=f'Cluster {label}')
        # Reference window points
        if reference_windows is not None:
            n_windows = len(reference_windows)
            for i, (cid, window_info) in enumerate(reference_windows.items()):
                color_idx = np.where(unique_labels == cid)[0][0]
                ref_color = ref_colors[color_idx]
                
                # Obtain the pca scores of window points
                window_points = window_info['window_points']
                window_scores = scores[window_points]
                
                
                # mark the window points
                ax.scatter(window_scores[:, 0], window_scores[:, 1], window_scores[:, 2],
                        s=80, marker='s', facecolor=ref_color, edgecolor='k',
                        linewidth=1.5, zorder=8, 
                        label=f'Ref Window (C{cid})')
                
                # Window center point
                center_idx = np.where(np.array(window_points) == window_info['center_indice'])[0]
                if len(center_idx) > 0:
                    center_scores = window_scores[center_idx]
                    label = 'Window Center' if i == n_windows - 1 else ""
                    ax.scatter(center_scores[:, 0], center_scores[:, 1], center_scores[:, 2],
                            s=100, marker='D', facecolor='red', edgecolor='k',
                            linewidth=1.5, zorder=9, 
                            label=label)
        
        if len(center_points) > 0:
            ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2],
                    s=300, marker='*', facecolor='gold', edgecolor='k', 
                    linewidth=1.5, zorder=10, label='Cluster Centers')
        # plot anomalies
        if anomalies is not None:
            anomalies = np.asarray(anomalies)
            assert anomalies.shape[1] >= 3, "anomalies dim must be higher than 3"
            ax.scatter(anomalies[:, 0], anomalies[:, 1], anomalies[:, 2],
                    s=80, marker='X', c='none', 
                    edgecolor='purple', label='Anomalies', linewidths=1.5)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        # ax.set_title(f'{title_prefix} for PCA scores (n_clusters={optimal_n})')

    else:  # dim == 2
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, label in enumerate(unique_labels):
            mask = (cluster_labels == label)
            ax.scatter(scores[mask, 0], 
                    scores[mask, 1], 
                    color=colors[i], alpha=0.7, label=f'Cluster {label}', edgecolors='k')
            _add_confidence_ellipse(ax, scores[mask, :2], colors[i], alpha=ellipse_alpha)
        # Reference window points
        if reference_windows is not None:
            n_windows = len(reference_windows)
            for i, (cid, window_info) in enumerate(reference_windows.items()):
                
                color_idx = np.where(unique_labels == cid)[0][0]
                ref_color = ref_colors[color_idx]
                
                
                window_points = window_info['window_points']
                window_scores = scores[window_points]
                
                
                ax.scatter(window_scores[:, 0], window_scores[:, 1],
                        s=80, marker='s', facecolor=ref_color, edgecolor='k',
                        linewidth=1.5, zorder=8, 
                        label=f'Ref Window (C{cid})')
                
                
                center_idx = np.where(np.array(window_points) == window_info['center_indice'])[0]
                if len(center_idx) > 0:
                    center_scores = window_scores[center_idx]
                    label = 'Window Center' if i == n_windows - 1 else ""
                    ax.scatter(center_scores[:, 0], center_scores[:, 1],
                            s=100, marker='D', facecolor='red', edgecolor='k',
                            linewidth=1.5, zorder=9, 
                            label=label)    
                    
        if len(center_points) > 0:
            ax.scatter(center_points[:, 0], center_points[:, 1],
                    s=300, marker='*', facecolor='gold', edgecolor='k', 
                    linewidth=1.5, zorder=10, label='Cluster Centers')
        # plot anomalies
        if anomalies is not None:
            anomalies = np.asarray(anomalies)
            assert anomalies.shape[1] >= 2, "anomalies dim must be higher than 2"
            ax.scatter(anomalies[:, 0], anomalies[:, 1], 
                    s=80, marker='X', c='none', 
                    edgecolor='purple', label='Anomalies', linewidths=1.5)

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        # ax.set_title(f'{title_prefix} for PCA scores (n_clusters={optimal_n})')
    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = {}
    for handle, label in zip(handles, labels):
        unique_labels_dict[label] = handle
    
    # Create ordered legend
    ordered_labels = []
    # Add cluster labels first
    for label in unique_labels:
        cluster_label = f'Cluster {label}'
        if cluster_label in unique_labels_dict:
            ordered_labels.append(cluster_label)
    
    # Add reference window labels
    if reference_windows is not None:
        for cid in reference_windows.keys():
            ref_label = f'Ref Window (C{cid})'
            if ref_label in unique_labels_dict:
                ordered_labels.append(ref_label)
    
    # Add other labels
    other_labels = ['Window Center', 'Cluster Centers', 'Anomalies']
    for label in other_labels:
        if label in unique_labels_dict:
            ordered_labels.append(label)
    
    # Create ordered handles
    ordered_handles = [unique_labels_dict[label] for label in ordered_labels]
    
    # Draw legend
    plt.legend(ordered_handles, ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    saved = _safe_savefig(fig, save_dir, filename, dpi)
    if show: plt.show()
    else: plt.close(fig)
    return saved

def _to_jsonable(obj):
    """
    Convert numpy/pandas objects recursively into JSON-serializable Python types.
    This version aggressively normalizes all numpy scalar types via np.generic.
    """
    # 1) Any NumPy scalar (np.int64, np.float32, np.bool_ ...) → Python scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # 2) Plain Python primitives
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # 3) Common NumPy containers
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 4) Pandas containers
    if isinstance(obj, pd.DataFrame):
        # More generalized
        return [_to_jsonable(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return _to_jsonable(obj.to_dict())

    # 5) Mappings / Sequences
    if isinstance(obj, dict):
        
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        
        return [_to_jsonable(v) for v in sorted(obj)]

    # 6) Fallback: try __dict__ / repr
    try:
        return _to_jsonable(obj.__dict__)
    except Exception:
        return repr(obj)


def save_evaluation_json(out, save_json_path):
    """Save evaluation result dict (from evaluate_clustering_metrics) to JSON safely."""
    payload = {
        "metrics": _to_jsonable(out.get("metrics", {})),
        "user_mapping": _to_jsonable(out.get("user_mapping", {})),
        "mapping": _to_jsonable(out.get("mapping", {})),
        "mapping_names": _to_jsonable(out.get("mapping_names", {})),
        "confusion_matrix": _to_jsonable(out.get("confusion_matrix", [])),
        "detailed_results": _to_jsonable(out.get("detailed_results", {})),
    }
    os.makedirs(os.path.dirname(save_json_path) or ".", exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def evaluate_clustering_metrics(coord_dict, coord_to_label, name_map, cluster_name_map=None, print_table=True, mapping_mode="majority", save_json_path: str | None = None):
    """
    Comprehensive clustering evaluation with configurable label mapping:
      - "majority": map each cluster to the true phase with the highest count in that cluster
      - "hungarian": optimal one-to-one mapping via Hungarian algorithm (requires scipy)

    Args
    ----
    coord_dict : dict
        (x, y) -> true phase index
    coord_to_label : dict
        (x, y) -> predicted cluster index
    name_map : dict
        phase index -> phase name
    cluster_name_map : dict or None
        cluster index -> phase name (user-specified). If None/empty, no user mapping is applied.
    print_table : bool
        whether to print metrics and tables
    mapping_mode : str
        "majority" (default) or "hungarian"

    Returns
    -------
    dict with:
        - metrics (incl. ARI, NMI, etc., user_accuracy if available, mapped accuracy)
        - mapping (chosen mapping dict: cluster -> phase index)
        - optimal_mapping (same as mapping, kept for backward compatibility)
        - confusion_matrix (after applying chosen mapping)
        - detailed_results (per-coordinate dataframe)
    """
    # 1) Shared coordinates
    shared_coords = sorted(set(coord_dict.keys()) & set(coord_to_label.keys()))
    if not shared_coords:
        raise ValueError("No shared coordinates between coord_dict and coord_to_label.")

    # 2) Extract labels
    y_true = [coord_dict[c] for c in shared_coords]
    y_pred = [coord_to_label[c] for c in shared_coords]

    # 3) Optional user mapping accuracy (cluster_name_map maps to NAMES, not indices)
    user_accuracy = None
    if cluster_name_map:
        y_pred_user_mapped = []
        valid_indices = []

        # Build reverse name_map: name -> index
        name_to_index = {v: k for k, v in name_map.items()}

        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            user_phase_name = cluster_name_map.get(pred_label, None)
            if user_phase_name is None:
                continue
            user_mapped_value = name_to_index.get(user_phase_name, None)
            if user_mapped_value is None:
                continue
            y_pred_user_mapped.append(user_mapped_value)
            valid_indices.append(i)

        if valid_indices:
            user_accuracy = accuracy_score(
                [y_true[i] for i in valid_indices],
                y_pred_user_mapped
            )

    # 4) Build mapping per mode

    def _contingency_true_by_pred(y_true_, y_pred_, unique_true=None, unique_pred=None):
        """
        Build a contingency matrix with rows = unique_true, cols = unique_pred.
        """
        if unique_true is None:
            unique_true = np.unique(y_true_)
        if unique_pred is None:
            unique_pred = np.unique(y_pred_)

        r = len(unique_true)
        c = len(unique_pred)
        cm = np.zeros((r, c), dtype=int)

        # map labels to indices
        t2i = {t: i for i, t in enumerate(unique_true)}
        p2j = {p: j for j, p in enumerate(unique_pred)}

        for t, p in zip(y_true_, y_pred_):
            # skip unseen labels defensively
            if t in t2i and p in p2j:
                cm[t2i[t], p2j[p]] += 1
        return cm, unique_true, unique_pred

    def _majority_mapping(y_true_, y_pred_):
        """
        For each predicted cluster (column), pick the true label (row) with max count.
        """
        cm, unique_true, unique_pred = _contingency_true_by_pred(y_true_, y_pred_)
        mapping = {}
        for col_idx, pred_label in enumerate(unique_pred):
            col_counts = cm[:, col_idx]
            if col_counts.sum() == 0:
                mapping[pred_label] = -1
            else:
                best_row = int(np.argmax(col_counts))
                mapping[pred_label] = int(unique_true[best_row])
        return mapping, cm, unique_true, unique_pred

    def _hungarian_mapping(y_true_, y_pred_):
        """
        Optimal 1-1 mapping via Hungarian on a (pred x true) cost matrix.
        """
        from scipy.optimize import linear_sum_assignment  # raise if unavailable

        # rows=true, cols=pred
        cm_true_by_pred, unique_true, unique_pred = _contingency_true_by_pred(y_true_, y_pred_)
        # Convert to (pred x true) so each row is a predicted cluster we must assign
        cm_pred_by_true = cm_true_by_pred.T  # shape: (len(unique_pred), len(unique_true))

        # maximize counts => minimize negative counts
        row_ind, col_ind = linear_sum_assignment(-cm_pred_by_true)

        mapping = {}
        for r, c in zip(row_ind, col_ind):
            pred_label = unique_pred[r]
            true_label = unique_true[c]
            mapping[pred_label] = true_label

        # return the contingency in rows=true, cols=pred convention (useful for inspection)
        return mapping, cm_true_by_pred, unique_true, unique_pred

    if mapping_mode.lower() == "hungarian":
        mapping, cm_raw, unique_true, unique_pred = _hungarian_mapping(y_true, y_pred)
        mapping_name = "Hungarian (optimal 1-1)"
    else:
        mapping, cm_raw, unique_true, unique_pred = _majority_mapping(y_true, y_pred)
        mapping_name = "Majority-vote"
    # 5) Apply chosen mapping
    y_pred_mapped = [mapping.get(lbl, -1) for lbl in y_pred]

    # 6) Metrics
    metrics = {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "v_measure": v_measure_score(y_true, y_pred),
        "mapped_accuracy": accuracy_score(y_true, y_pred_mapped),
        "mapping_mode": mapping_name,
    }
    metrics["user_accuracy"] = user_accuracy

    # 7) Per-coordinate details
    results = []
    name_to_index = {v: k for k, v in name_map.items()}
    for coord, true_label, pred_label in zip(shared_coords, y_true, y_pred):
        true_name = name_map.get(true_label, f"Phase_{true_label}")

        # user mapping (if provided)
        user_phase_name = cluster_name_map.get(pred_label, "Unknown") if cluster_name_map else "—"
        user_mapped_value = name_to_index.get(user_phase_name) if cluster_name_map else None
        user_match = (true_label == user_mapped_value) if user_mapped_value is not None else False

        # chosen mapping
        mapped_label = mapping.get(pred_label, -1)
        mapped_name = name_map.get(mapped_label, "Unmapped")
        is_match = (true_label == mapped_label)

        results.append({
            "coordinate": coord,
            "true_phase": true_name,
            "pred_cluster": pred_label,
            "user_mapped_phase": user_phase_name,
            "user_match": user_match,
            "mapped_phaseid": mapped_label,
            "mapped_phase": mapped_name,
            "match": is_match
        })

    df = pd.DataFrame(results)

    # 8) Pretty print
    if print_table:
        print("=" * 80)
        print("Clustering Evaluation Results")
        print("=" * 80)

        print("\nEvaluation Metrics:")
        print("-" * 60)
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        print(metrics_df.to_string(index=False))

        print(f"\nCluster-to-Phase Mapping ({mapping_name}):")
        print("-" * 60)

        # All clusters that appear anywhere
        all_clusters = sorted(set(coord_to_label.values()) | set(mapping.keys()))
        mapping_rows = []
        for cl in all_clusters:
            user_map_name = cluster_name_map.get(cl, "—") if cluster_name_map else "—"
            mapped_idx = mapping.get(cl, -1)
            mapped_name = name_map.get(mapped_idx, "Unmapped")
            cluster_size = sum(1 for lbl in y_pred if lbl == cl)

            mapping_rows.append({
                "cluster": cl,
                "Samples": cluster_size,
                "User mapping": user_map_name,
                "Chosen mapping": mapped_name
            })
        print(pd.DataFrame(mapping_rows).to_string(index=False))

        print("\nConfusion Matrix (After Mapping):")
        print("-" * 60)
        all_labels = sorted(set(y_true) | set(y_pred_mapped))
        conf_df = pd.DataFrame(
            confusion_matrix(y_true, y_pred_mapped, labels=all_labels),
            index=[f"True: {name_map.get(i, f'Phase_{i}')}" for i in all_labels],
            columns=[f"Pred: {name_map.get(j, f'Phase_{j}')}" for j in all_labels]
        )
        print(conf_df)

    # 9) Assemble return dict
    out = {
        "metrics": metrics,
        "user_mapping": (cluster_name_map if cluster_name_map else {}),
        "mapping": mapping,  # cluster_id -> phase_id
        "optimal_mapping": mapping,
        "mapping_names": {int(cid): name_map.get(int(pid), "Unknown") for cid, pid in mapping.items()},
        "confusion_matrix": confusion_matrix(y_true, y_pred_mapped),
        "detailed_results": df,
    }

    # 10) saved as JSON
    if save_json_path:
        save_evaluation_json(out, save_json_path=save_json_path)

    return out
    

def compute_mahalanobis_to_reference(pca_scores, loc, cluster_labels, ref_coord_dict):
    """
    For each point, compute Mahalanobis distance to its cluster's reference point in PCA space.

    Args:
        pca_scores (ndarray): shape (n_samples, n_components)
        loc (ndarray): shape (n_samples, 2), coordinates for each sample in the same order
        cluster_labels (ndarray): shape (n_samples,), integer cluster assignments
        ref_coord_dict (dict): mapping cluster_label -> (x, y) reference coordinate

    Returns:
        distances (ndarray): shape (n_samples,), Mahalanobis distances for each sample in loc order
    """
    # Build coord->index map
    loc_array = np.asarray(loc)
    coord_to_index = {tuple(loc_array[i]): i for i in range(loc_array.shape[0])}
    
    # Initialize distance array with NaNs (for samples without valid reference)
    distances = np.full(loc_array.shape[0], np.nan)
    
    # Precompute cluster covariances and reference vectors
    cluster_info = {}
    for cluster in np.unique(cluster_labels):
        # Get reference point for this cluster
        ref_coord = ref_coord_dict.get(cluster)
        if ref_coord is None:
            continue
        
        ref_idx = coord_to_index.get(tuple(ref_coord))
        if ref_idx is None:
            print(f"Warning: Reference coordinate {ref_coord} not found in loc array for cluster {cluster}")
            continue
        
        # Get points in this cluster
        mask = cluster_labels == cluster
        cluster_scores = pca_scores[mask]
        
        # Only compute covariance if we have enough points
        if len(cluster_scores) < 2:
            print(f"Warning: Cluster {cluster} has only {len(cluster_scores)} points. Cannot compute covariance.")
            continue
        
        # Compute covariance and its inverse
        cov = np.cov(cluster_scores, rowvar=False)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            inv_cov = np.linalg.pinv(cov)
            print(f"Warning: Covariance matrix for cluster {cluster} is singular. Using pseudo-inverse.")
        
        cluster_info[cluster] = {
            'ref_vec': pca_scores[ref_idx],
            'inv_cov': inv_cov,
            'mask': mask
        }
    
    # Compute distances for each cluster
    for cluster, info in cluster_info.items():
        ref_vec = info['ref_vec']
        inv_cov = info['inv_cov']
        mask = info['mask']
        
        # Get scores for this cluster
        cluster_scores = pca_scores[mask]
        
        # Compute differences to reference
        diffs = cluster_scores - ref_vec
        
        # Mahalanobis distance computation using the vector methods
        quad_form = np.sum(diffs @ inv_cov * diffs, axis=1)
        dists = np.sqrt(quad_form)
        
        # Assign distances to the correct positions in the result array
        distances[mask] = dists
    
    return distances