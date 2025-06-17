from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, Colormap, to_rgba
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Patch, Ellipse
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from PCA import _add_confidence_ellipse

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
        cluster_labels (np.array): clustering labels
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
                                    default_cluster_cmap_names=None, cluster_name_map=None, anomalies_cluster_pca_coords3=None):
    """
    Plots a heatmap showing intra-cluster variation. Each cluster/phase uses its
    own colormap to highlight its internal variations.
    
    Args:
        loc_roi (array): coordinates
        variation (array):  Variation score (e.g., Mahalanobis dist) for each sample.
        cluster_labels (array): Cluster assignment for each sample.
        cluster_coords (dict): Coordinates grouped by labels
        img_shape (tuple)
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
        print("Cluster centers and corresponding labels:", min_variation_points)
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
                                linewidth=2, edgecolor='gray', facecolor='none')
            ax.add_patch(rect)
            min_var_patches.append(rect)
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
        center_legend = Patch(facecolor='none', edgecolor='gray', linewidth=2, label='"Central" Point')
        legend_handles.append(center_legend)
    if anomaly_patch_handles:
        anomaly_legend = Patch(facecolor='none', edgecolor='black', linewidth=2, label='Anomaly Point')
        legend_handles.append(anomaly_legend)

    if legend_handles:
        fig.legend(handles=legend_handles,
                loc='lower center',
                bbox_to_anchor=(0.7, 0.02),
                ncol=len(legend_handles),
                frameon=False)

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

def plot_gmm_clusters(pca_scores, cluster_labels, optimal_n, variations, dim=2, anomalies=None, title_prefix="GMM Clustering", ellipse_alpha=0.3):
    """
    Plot GMM clustering results in 2D or 3D based on selected PCA components.

    Args:
        pca_scores (ndarray): PCA transformed data, shape (n_samples, n_components)
        cluster_labels (ndarray): Cluster labels
        optimal_n (int): Optimal number of clusters
        variations (ndarray): Mahalanobis distance of each sample within one cluster
        dim (int): 2 or 3, to plot in 2D or 3D space using first dim principal components
        anomalies (ndarray or None): Optional anomalies to be marked, shape (n_anomalies, 2)
        title_prefix (str): Plot title prefix
        ellipse_alpha (float): Transparency for confidence ellipses
    """
    assert dim in [2, 3], "Only dim=2 or dim=3 is supported."
    assert pca_scores.shape[1] >= dim, f"pca_scores must have at least {dim} components."

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
        center_points.append(pca_scores[global_idx])
        center_labels.append(label)
    center_points = np.array(center_points)
    num_clusters = len(unique_labels)
    default_cluster_cmap_names = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 
                                'YlOrBr', 'BuGn', 'PuRd', 'Greys'][:num_clusters]
    colors = [cm.get_cmap(name)(0.6) for name in default_cluster_cmap_names]

    if dim == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            mask = (cluster_labels == label)
            ax.scatter(pca_scores[mask, 0], 
                    pca_scores[mask, 1], 
                    pca_scores[mask, 2],
                    color=colors[i], alpha=0.7, label=f'Cluster {label}')
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
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'{title_prefix} (n_clusters={optimal_n})')

    else:  # dim == 2
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, label in enumerate(unique_labels):
            mask = (cluster_labels == label)
            ax.scatter(pca_scores[mask, 0], 
                    pca_scores[mask, 1], 
                    color=colors[i], alpha=0.7, label=f'Cluster {label}', edgecolors='k')
            _add_confidence_ellipse(ax, pca_scores[mask, :2], colors[i], alpha=ellipse_alpha)
            
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

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{title_prefix} (n_clusters={optimal_n})')
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def clustering_accuracy(coord_dict, coord_to_label, name_map, cluster_name_map):
    """
    Compare the accuracy of clustering labels with phase index

    Args:
        coord_dict: dict, (x, y) -> phase index
        coord_to_label: dict, (x, y) -> cluster index
        name_map: dict, phase index -> phase name
        cluster_name_map: dict, cluster index -> phase name or "Unknown"

    Returns:
        match_results: dict, (x, y): True/False
        accuracy: float
    """
    match_results = {}
    n_matched = 0
    n_total = 0

    # The common points/COORDINATES
    shared_coords = set(coord_dict.keys()) & set(coord_to_label.keys())
    for coord in shared_coords:
        phase_idx = coord_dict[coord]
        cluster_idx = coord_to_label[coord]
        phase_name = name_map.get(phase_idx, "Unknown")
        cluster_phase_name = cluster_name_map.get(cluster_idx, "Unknown")
        is_match = (phase_name == cluster_phase_name)
        match_results[coord] = is_match
        n_matched += int(is_match)
        n_total += 1

    accuracy = n_matched / n_total if n_total > 0 else 0.0
    return match_results, accuracy



