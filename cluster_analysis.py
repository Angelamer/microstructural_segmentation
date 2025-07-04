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

# from scipy.spatial.distance import mahalanobis


def gmm_clustering(scores, loc_roi, n_components=None, max_components=10, random_state=42):
    """
    GMM clustering for pca scores
    
    Args:
        scores (np.array): data (n_samples, n_components)
        loc_roi : lists of coordinates
        n_components (int/None): predefined clustering number
        max_components (int): max clustering number when automatically searching 
        random_state (int)
    
    Return:
        gmm: models
        cluster_labels (np.array): clustering labels
        cluster_coords (dict): the dict of cluster label as keys with coordinates as values
        optimal_n (int): 
        silhouette (float): Silhouette Coefficient
    """
    loc_roi = np.asarray(loc_roi)
    assert len(scores) == len(loc_roi), "The sample number should be consistent."

    # find optimal clustering number 
    if n_components is None:
        n_components = np.arange(2, max_components+1)
        silhouettes = []
        models = []
        for n in n_components:
            gmm = GaussianMixture(n_components=n, random_state=random_state)
            labels = gmm.fit_predict(scores)
            sil = silhouette_score(scores, labels)
            silhouettes.append(sil)
            models.append((gmm, labels))

        optimal_index = np.argmax(silhouettes)
        optimal_n = n_components[optimal_index]
        gmm, cluster_labels = models[optimal_index]
        silhouette = silhouettes[optimal_index]
    else:
        optimal_n = n_components
    
        # Gaussian mixture
        gmm = GaussianMixture(n_components=optimal_n, random_state=random_state)
        cluster_labels = gmm.fit_predict(scores)
        silhouette = silhouette_score(scores, cluster_labels) if optimal_n > 1 else np.nan
    
    # Categorized by the labels
    cluster_coords = {}
    for label in np.unique(cluster_labels):
        mask = (cluster_labels == label)
        cluster_coords[label] = loc_roi[mask].tolist()
    
    return gmm, cluster_coords, cluster_labels, optimal_n, silhouette

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

def plot_cluster_distances_ranking(gmm_model, cluster_labels, scores, loc, top_n=10):
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
    plt.figure(figsize=(12, 8))

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
    
    plt.xlabel('Sample Rank (sorted by Mahalanobis distance)')
    plt.ylabel('Mahalanobis Distance')
    plt.title('Mahalanobis Distance Ranking by Cluster')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print the information
    print("Top samples per cluster:")
    for label, info in top_samples_per_cluster.items():
        print(f"Cluster {label}: {len(info['coordinates'])} samples")
        print(f"  Distance range: {info['distances'][0]:.4f} - {info['distances'][-1]:.4f}")
        print(f"  Coordinates: {info['coordinates'][:3]}...")  # 显示前3个坐标
        print()
    
    return top_samples_per_cluster

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
def plot_cluster_heatmap(cluster_coords, img_shape=(31, 31)):
    
    heatmap = np.zeros(img_shape, dtype=int) -1
    
    # color mapping
    unique_labels = sorted(cluster_coords.keys())
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    cmap = ListedColormap(colors(np.arange(len(unique_labels))))
    
    for label, coords in cluster_coords.items():
        for (x, y) in coords:
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                heatmap[y, x] = label
    if -1 in unique_labels:
        n_colors = len(unique_labels)
        color_values = plt.cm.tab10(np.linspace(0, 1, n_colors))
        color_values[0] = [0.7, 0.7, 0.7, 1] # if label=-1, gray
    else:
        n_colors = len(unique_labels)
        color_values = plt.cm.tab10(np.linspace(0, 1, n_colors))
    #color mapping
    cmap = ListedColormap(color_values)
    norm = plt.Normalize(vmin=min(unique_labels)-0.5, 
                        vmax=max(unique_labels)+0.5)
    
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heatmap, cmap=cmap, interpolation='nearest', 
                norm=norm)
    legend_elements = [
        Patch(facecolor=color_values[i], 
        edgecolor='k', 
        label=f'Cluster {label}' if label != -1 else 'Noise')
        for i, label in enumerate(unique_labels)
    ]
    plt.legend(handles=legend_elements, 
            bbox_to_anchor=(1, 1), 
            loc='upper left',
            title='Clusters')
    plt.title("Cluster Distribution Heatmap")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_intra_cluster_variation_map(loc_roi, variation, cluster_labels, img_shape=(31, 31),
                                    default_cluster_cmap_names=None, cluster_name_map=None, anomalies_cluster_pca_coords3=None, ref1_pos=None, ref2_pos=None, reference_windows=None):
    """
    Plots a heatmap showing intra-cluster variation. Each cluster/phase uses its
    own colormap to highlight its internal variations.
    
    Args:
        loc_roi (array): coordinates
        variation (array):  Variation score (e.g., Mahalanobis dist) for each sample.
        cluster_labels (array): Cluster assignment for each sample.
        cluster_coords (dict): Coordinates grouped by labels
        img_shape (tuple)
        reference_windows (dict): Reference windows from find_best_window
    """
    
    loc_roi = np.asarray(loc_roi)
    
    if not (len(loc_roi) == len(cluster_labels) == len(variation)):
        raise ValueError("All input arrays must have the same length.")
    
    if len(loc_roi) == 0:
        print("No data to plot.")
        return
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
            x_int, y_int = int(round(x)), int(round(y))
            if 0 <= x_int < img_shape[1] and 0 <= y_int < img_shape[0]:
                norm_val = normalized_variations[idx]
                if not np.isnan(norm_val):
                    color_rgba = current_cmap(norm_val)
                    rgba_heatmap[y_int, x_int, :] = color_rgba
    print("Cluster centers and corresponding labels:", min_variation_points)
    # Display the RGBA heatmap
    ax.imshow(rgba_heatmap, interpolation='nearest', origin='upper')
    ax.set_title("Intra-Cluster Variation (Mahalanobis Distance)", fontsize=16)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # Box select the point with the min distance
    min_var_patches = []
    for x, y, label_val in min_variation_points:
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= x_int < img_shape[1] and 0 <= y_int < img_shape[0]:
            # Create a rectangle
            rect = plt.Rectangle((x_int - 0.5, y_int - 0.5), 1, 1, 
                                linewidth=3, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            min_var_patches.append(rect)
    # Reference points box
    ref_win_patches = []
    if reference_windows is not None:
        for cid, win_info in reference_windows.items():
            # obtain the window point indices
            window_points = win_info['window_points']
            for idx in window_points:
                x, y = loc_roi[idx]
                x_int, y_int = int(round(x)), int(round(y))
                if 0 <= x_int < img_shape[1] and 0 <= y_int < img_shape[0]:
                    rect = plt.Rectangle((x_int - 0.5, y_int - 0.5), 1, 1, 
                                        linewidth=1.5, edgecolor='red', 
                                        facecolor='none', hatch='////', alpha=0.8)
                    ax.add_patch(rect)
                    ref_win_patches.append(rect)
    ref1_patches = []
    ref2_patches = []
    # Plot ref1_pos with cluster colors
    if ref1_pos is not None:
        for x, y in ref1_pos:
            x_int, y_int = int(round(x)), int(round(y))
            if 0 <= x_int < img_shape[1] and 0 <= y_int < img_shape[0]:
                # Find cluster label for this point (nearest neighbor)
                distances = np.linalg.norm(loc_roi - [x, y], axis=1)
                nearest_idx = np.argmin(distances)
                cluster_label = cluster_labels[nearest_idx]
                    
                if cluster_label in cluster_cmaps:
                    color = cluster_cmaps[cluster_label](0.5)  # Mid-point color
                    rect = plt.Rectangle((x_int - 0.5, y_int - 0.5), 1, 1, 
                                        linewidth=2, edgecolor=color, 
                                        facecolor='none', linestyle='-')
                    ax.add_patch(rect)
                    ref1_patches.append(rect)
        
    # Plot ref2_pos with cluster colors
    if ref2_pos is not None:
        for x, y in ref2_pos:
            x_int, y_int = int(round(x)), int(round(y))
            if 0 <= x_int < img_shape[1] and 0 <= y_int < img_shape[0]:
                # Find cluster label for this point (nearest neighbor)
                distances = np.linalg.norm(loc_roi - [x, y], axis=1)
                nearest_idx = np.argmin(distances)
                cluster_label = cluster_labels[nearest_idx]
                    
                if cluster_label in cluster_cmaps:
                    color = cluster_cmaps[cluster_label](0.5)  # Mid-point color
                    rect = plt.Rectangle((x_int - 0.5, y_int - 0.5), 1, 1, 
                                        linewidth=2, edgecolor=color, 
                                        facecolor='none', linestyle='--')  # Dashed for distinction
                    ax.add_patch(rect)
                    ref2_patches.append(rect)
    anomaly_patch_handles = []
    if anomalies_cluster_pca_coords3 is not None:
        anomalies_cluster_pca_coords3 = np.asarray(anomalies_cluster_pca_coords3)
        for x, y in anomalies_cluster_pca_coords3:
            x_int, y_int = int(round(x)), int(round(y))
            if 0 <= x_int < img_shape[1] and 0 <= y_int < img_shape[0]:
                rect = plt.Rectangle((x_int - 0.5, y_int - 0.5), 1, 1,
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
    plt.show()
    

    
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
                                ref1_pos, ref2_pos, d=0.1, title_prefix="GMM Clustering for cNMF weights", 
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
    
    # Boundary points
    distances = np.abs(weights[:, 0] - weights[:, 1]) / np.sqrt(2)
    boundary_mask = distances < d
    boundary_weights = weights[boundary_mask]
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
    
    return boundary_mask, boundary_scores, boundary_locs, slope, intercept


def plot_gmm_clusters(scores, cluster_labels, optimal_n, variations, dim=2, anomalies=None, reference_windows=None, title_prefix="GMM Clustering", ellipse_alpha=0.3):
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
                                'YlOrBr', 'BuGn', 'PuRd', 'Greys'][:num_clusters]
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
        ax.set_title(f'{title_prefix} for PCA scores (n_clusters={optimal_n})')

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
        ax.set_title(f'{title_prefix} for PCA scores (n_clusters={optimal_n})')
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
    plt.show()


def evaluate_clustering_metrics(coord_dict, coord_to_label, name_map, cluster_name_map, print_table=True):
    """
    Comprehensive clustering evaluation with optimal label mapping and confusion matrix
    
    Args:
        coord_dict: dict, (x, y) -> phase index
        coord_to_label: dict, (x, y) -> cluster index
        name_map: dict, phase index -> phase name
        cluster_name_map: dict, cluster index -> phase name or "Unknown"
        print_table: bool, whether to print results
        
    Returns:
        dict with evaluation metrics, mapping, and confusion matrix
    """
    # Get shared coordinates
    shared_coords = sorted(set(coord_dict.keys()) & set(coord_to_label.keys()))

    
    # Extract true and predicted labels
    y_true = [coord_dict[c] for c in shared_coords]
    y_pred = [coord_to_label[c] for c in shared_coords]
    
    # User-defined mapping
    y_pred_user_mapped = []
    valid_indices = []
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        # Obtain the user mapping phase name
        user_phase_name = cluster_name_map.get(pred_label, "Unknown")
        
        # Search the corresponding value label
        user_mapped_value = None
        for k, v in name_map.items():
            if v == user_phase_name:
                user_mapped_value = k
                break

        if user_mapped_value is not None:
            y_pred_user_mapped.append(user_mapped_value)
            valid_indices.append(i)
    
    # Onlu caculate the valid mapping accuracy
    if valid_indices:
        user_accuracy = accuracy_score(
            [y_true[i] for i in valid_indices],
            [y_pred_user_mapped[i] for i in range(len(valid_indices))]
        )
    else:
        user_accuracy = 0.0
        
    # Create label mapping using Hungarian algorithm
    def _create_optimal_mapping(y_true, y_pred):
        # Create confusion matrix
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=unique_true)
        
        # Apply Hungarian algorithm for optimal mapping
        row_ind, col_ind = linear_sum_assignment(-cm)
        
        # Create mapping dictionary
        mapping = {}
        for true_idx, pred_idx in zip(row_ind, col_ind):
            true_label = unique_true[true_idx]
            pred_label = unique_pred[pred_idx]
            mapping[pred_label] = true_label
        
        return mapping, cm
    
    # Create optimal mapping
    mapping, conf_matrix = _create_optimal_mapping(y_true, y_pred)
    
    # Apply mapping to predicted labels
    y_pred_mapped = [mapping.get(label, -1) for label in y_pred]
    
    # Calculate metrics
    metrics = {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "v_measure": v_measure_score(y_true, y_pred)
    }
    # Two kinds of accuracy and Hungarian Algorithm
    metrics["user_accuracy"] = user_accuracy
    metrics["optimal_accuracy"] = accuracy_score(y_true, y_pred_mapped)
    # Create detailed results table
    results = []
    for coord, true_label, pred_label in zip(shared_coords, y_true, y_pred):
        true_name = name_map.get(true_label, f"Phase_{true_label}")
        
        # user mapping
        user_phase_name = cluster_name_map.get(pred_label, "Unknown")
        user_mapped_value = None
        for k, v in name_map.items():
            if v == user_phase_name:
                user_mapped_value = k
                break
        user_match = (true_label == user_mapped_value) if user_mapped_value is not None else False
        
        # Hungarian mapping
        mapped_label = mapping.get(pred_label, -1)
        mapped_name = name_map.get(mapped_label, "Unmapped")
        is_match = (true_label == mapped_label)
        
        results.append({
            "coordinate": coord,
            "true_phase": true_name,
            "pred_cluster": pred_label,
            "user_mapped_phase": user_phase_name,
            "user_match": user_match,
            "mapped_phase": mapped_name,
            "match": is_match
        })
    
    df = pd.DataFrame(results)
    
    # Print results if requested
    if print_table:
        print("="*80)
        print("Clustering Evaluation Results")
        print("="*80)
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print("-"*60)
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        print(metrics_df.to_string(index=False))
        
        # Print mapping
        print("\nOptimal Cluster-to-Phase Mapping:")
        print("-"*60)
        
        # All labels
        all_clusters = set(coord_to_label.values()) | set(cluster_name_map.keys()) | set(mapping.keys())
        
        mapping_data = []
        for cluster in sorted(all_clusters):
            user_map = cluster_name_map.get(cluster, "Unknown")
            
            optimal_value = mapping.get(cluster, -1)
            optimal_map = name_map.get(optimal_value, "Unmapped")
            
            cluster_size = sum(1 for lbl in y_pred if lbl == cluster)
            
            mapping_data.append({
                "cluster": cluster,
                "Samples": cluster_size,
                "User mapping": user_map,
                "Algorithm mapping": optimal_map
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        print(mapping_df.to_string(index=False))
        
        print("\nConfusion Matrix (After Mapping):")
        print("-"*60)
        
        all_labels = sorted(set(y_true) | set(y_pred_mapped))
        
        
        # Confusion matrix for the Hungarian mapping
        conf_df = pd.DataFrame(
            confusion_matrix(y_true, y_pred_mapped, labels=all_labels),
            index=[f"True: {name_map.get(i, f'Phase_{i}')}" for i in all_labels],
            columns=[f"Pred: {name_map.get(j, f'Phase_{j}')}" for j in all_labels]
        )
        print(conf_df)

    
    return {
        "metrics": metrics,
        "user_mapping": cluster_name_map,
        "optimal_mapping": mapping,
        "confusion_matrix": confusion_matrix(y_true, y_pred_mapped),
        "detailed_results": df
    }