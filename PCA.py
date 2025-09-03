#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function to initiate PCA decomposition using the sklearn

Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_processing import signal_process
from progress.bar import Bar
import os
import cv2
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as ssim



def run_PCA(ROI,components,h,w,slice_x,slice_y):
    """
    Perform Principal Component Analysis (PCA) on processed Electron Backscatter Diffraction (EBSD) patterns.
    
    This function processes a list of EBSD pattern files, applies signal processing to each pattern,
    and then performs PCA dimensionality reduction on the processed patterns.
    
    Parameters:
    ----------
    ROI : list of str
        List of file paths pointing to EBSD pattern files to be processed and analyzed.
    
    components : int
        Number of principal components to compute and retain in the PCA transformation.
    
    h : int
        Height of the original EBSD patterns before any processing or slicing.
    
    w : int
        Width of the original EBSD patterns before any processing or slicing.
    
    slice_x : tuple of int
        Slice coordinates in x-direction as (start, end) for focusing on specific parts 
        of the Kikuchi patterns, typically to remove edge artifacts.
    
    slice_y : tuple of int
        Slice coordinates in y-direction as (start, end) for focusing on specific parts 
        of the Kikuchi patterns, typically to remove edge artifacts.
    
    Returns:
    -------
    pca_scores : numpy.ndarray
        PCA transformed data with shape (n_samples, n_components), where n_samples is the 
        number of processed patterns and n_components is the specified number of principal components.
    
    pca : sklearn.decomposition.PCA
        Fitted PCA model object that can be used for further transformations or analysis.
        This object contains attributes such as explained_variance_ratio_, components_, etc.
    """
    
    file_list = ROI
    bar = Bar("Processing", max=len(file_list))
    signal_processed = []
    
    for file in file_list:
        bar.next()
        signal_processed.append(signal_process(file, flag="ROI",pattern_height=h, pattern_width=w, slice_x=slice_x, slice_y=slice_y))  #31*31 EBSP with feature dimension of 150*150; But here is input_X 1*90000
        
    
    signal_processed = np.array(signal_processed).reshape(len(file_list),-1)
    
    # print(np.shape(signal_processed))
    pca = PCA(n_components = components)
    pca_scores = pca.fit_transform(signal_processed)

    bar.finish()
    # print explained variance ratio

    print("Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)
    return pca_scores, pca

def plot_explained_variance(pca):
    """Plot the explained variance ratio and cumulative curve"""
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance)+1), explained_variance, 
            alpha=0.5, align='center', label='Individual Explained Variance')
    plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, 
            where='mid', label='Cumulative Explained Variance')
    
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

"""
def _plot_pca(pca_scores, coord_dict, loc_roi):
    
    loc_roi = np.asarray(loc_roi)
    assert loc_roi.shape[0] == pca_scores.shape[0], "make sure the number of samples/rows of pca scores the same with loc"
    assert loc_roi.shape[1] == 2, "loc should be list of (n_samples, 2)"
    
    roi_labels = [coord_dict.get((x, y), -1)
    for x, y in loc_roi]
    
    roi_labels = np.array(roi_labels)
    
    # corresponding phase mapping
    name_map = {
        1: 'Fe3O4',
        2: 'FeO',
        3: 'Fe'
        }
    
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=roi_labels, cmap='Set1', alpha=0.7,edgecolors='k')
    
    
    unique_ids = sorted(set(roi_labels.tolist()))
    
    # add legends
    handles = []
    for pid in unique_ids:
        if pid in name_map:
            color = scatter.cmap(scatter.norm(pid))
            patch = mpatches.Patch(color=color, label=name_map[pid])
            handles.append(patch)
    plt.legend(handles=handles, title='Phase')
    
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of EBSD Kikuchi Patterns by Phase index")
    plt.show()
"""
def _plot_reference(ref_pos, loc_roi, ax, pca_scores, marker, label, dim):
        if ref_pos is None or len(ref_pos) == 0:
            return
        
        ref_indices = []
        for pos in ref_pos:
            idx = np.where((loc_roi[:, 0] == pos[0]) & (loc_roi[:, 1] == pos[1]))[0]
            if len(idx) > 0:
                ref_indices.append(idx[0])
        
        if len(ref_indices) > 0:
            ref_pca = pca_scores[ref_indices]
            if dim == 3:
                ax.scatter(
                    ref_pca[:, 0], ref_pca[:, 1], ref_pca[:, 2],
                    s=150, marker=marker, c='gold', edgecolor='k',
                    linewidth=1.5, zorder=10, label=label
                )
            else:
                ax.scatter(
                    ref_pca[:, 0], ref_pca[:, 1],
                    s=150, marker=marker, c='gold', edgecolor='k',
                    linewidth=1.5, zorder=10, label=label
                )
            
def _add_confidence_ellipse(ax, data, color, alpha):
    """Add the confidence ellipse for each category"""
    if len(data) < 2: 
        return 
    
    # Compute the ellipse parameters
    cov = np.cov(data.T)
    try:
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(lambda_)
        chi = np.sqrt(chi2.ppf(0.95, 2))  # 95% confidence interval
        
        # Calculate ellipse angle
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
        
        # Plot the ellipse
        ell = Ellipse(
            xy=np.mean(data, axis=0),
            width=lambda_[0] * chi * 2, 
            height=lambda_[1] * chi * 2,
            angle=angle,
            edgecolor=color, 
            facecolor='none', 
            linestyle='--', 
            alpha=alpha
        )
        ax.add_patch(ell)
    except np.linalg.LinAlgError:
        # Handle cases where covariance matrix is singular
        pass
def _plot_pca(pca_scores, coord_dict, loc_roi, dim=2, ref1_pos=None, ref2_pos=None, anomalies=None, ellipse_alpha=0.3):
    """
    Plot PCA scatter map of samples with optional annotations.

    This function visualizes PCA scores (2D or 3D) of ROI samples and
    overlays phase labels, reference points, anomalies, and confidence ellipses
    for better interpretation of the data.

    Args:
        pca_scores (np.ndarray):
            Array of PCA-transformed data of shape (n_samples, n_components).
            Must have at least `dim` components.
        coord_dict (dict):
            Mapping from (x, y) coordinate tuples to phase IDs.
        loc_roi (array-like):
            ROI coordinates of shape (n_samples, 2).
            Each row is (x, y).
        dim (int, default=2):
            Number of PCA dimensions to plot (2 or 3).
        ref1_pos (list[tuple] or np.ndarray, optional):
            Coordinates of reference component 1 (highlighted with '*').
        ref2_pos (list[tuple] or np.ndarray, optional):
            Coordinates of reference component 2 (highlighted with 'P').
        anomalies (np.ndarray, optional):
            PCA coordinates of anomalous points to highlight with 'X'.
        ellipse_alpha (float, default=0.3):
            Transparency level for confidence ellipses (only applied in 2D).
        
    """
    assert dim in [2, 3]
    assert pca_scores.shape[1] >= dim, f"pca_scores must have at least {dim} components."
    
    loc_roi = np.asarray(loc_roi)
    assert loc_roi.shape[0] == pca_scores.shape[0], "make sure the number of samples/rows of pca scores the same with loc"
    assert loc_roi.shape[1] == 2, "loc should be list of (n_samples, 2)"
    
    roi_labels = [coord_dict.get((x, y), -1)
    for x, y in loc_roi]
    
    roi_labels = np.array(roi_labels)
    
    # corresponding phase mapping
    name_map = {
        1: 'Fe3O4',
        2: 'FeO',
        3: 'Fe'
        }
    phase_colors = {1: 'red', 2: 'blue', 3: 'green'}
    default_color = 'gray'
    
    if dim == 3:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(15, 10))
    colors = []
    for l in roi_labels:
        if l in phase_colors:
            colors.append(phase_colors[l])
        else:
            colors.append(default_color)
    
    # plot the main scatter
    if dim == 3:
        main_scatter = ax.scatter(
            pca_scores[:, 0], pca_scores[:, 1], pca_scores[:, 2],
            c=colors, alpha=0.7, edgecolors='k', label='Samples'
        )
    else:
        main_scatter = ax.scatter(
            pca_scores[:, 0], pca_scores[:, 1], 
            c=colors, alpha=0.7, edgecolors='k', label='Samples'
        )
    # mark the reference point
    _plot_reference(ref1_pos, loc_roi, ax, pca_scores, '*', 'Reference 1', dim=dim)
    _plot_reference(ref2_pos, loc_roi, ax, pca_scores, 'P', 'Reference 2', dim=dim)
    
    
    # plot the confidence ellipse (only for 2D)
    if dim == 2:
        for phase, color in phase_colors.items():
            mask = (roi_labels == phase)
            if mask.sum() > 1: 
                _add_confidence_ellipse(
                    ax, pca_scores[mask, :2], color, ellipse_alpha
                )
    
    # anomalies plotting
    if anomalies is not None:
        if dim ==3:
            ax.scatter(
                    anomalies[:, 0], anomalies[:, 1], anomalies[:, 2],
                    s=80, marker='X', c='none', 
                    edgecolor='purple', label='Anomalies', linewidths=1.5
                )
        else:
            ax.scatter(
                anomalies[:, 0], anomalies[:, 1], 
                s=80, marker='X', c='none', 
                edgecolor='purple', label='Anomalies',linewidths=1.5
            )
    # legend 
    legend_elements = []
    existing_phases = [pid for pid in np.unique(roi_labels) 
                    if pid in name_map and pid in phase_colors]
    
    for pid in sorted(existing_phases, key=lambda x: list(name_map.keys()).index(x)):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                label=name_map[pid],
                markerfacecolor=phase_colors[pid], 
                markersize=10)
        )
    if ref1_pos is not None and len(ref1_pos) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='*', color='k', label='Reference 1',
                markerfacecolor='none', markersize=15)
        )
    
    if ref2_pos is not None and len(ref2_pos) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='P', color='k', label='Reference 2',
                markerfacecolor='none', markersize=15)
        )
    if anomalies is not None and len(anomalies) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='X', color='purple', label='Anomalies',
                markerfacecolor='none', markersize=15)
        )
    if len([pid for pid in np.unique(roi_labels) if pid not in name_map]) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                label='Undefined Phase',
                markerfacecolor=default_color, 
                markersize=10)
        ) 
    
    if legend_elements:
        ax.legend(handles=legend_elements, title='Legend')
    ax.set_xlabel("Principal Component 1"), ax.set_ylabel("Principal Component 2")
    if dim == 3:
        ax.set_zlabel("Principal Component 3")
    ax.set_title("PCA Visualization with Annotations")
    plt.tight_layout()
    plt.show()

# detect the anomalies out of the confidence eclipse
def detect_anomalies_pca(pca_scores, coord_to_label, loc_roi):
    """
    Detect anomalies in PCA-transformed data using Mahalanobis distance.
    
    Anomalies are identified as points that lie outside the 95% confidence 
    ellipse of each manually defined phase/group. The phase category for each 
    point is obtained from `coord_to_label`.
    
    Parameters
    ----------
    pca_scores : np.ndarray
        Array of shape (n_samples, n_components) representing PCA-transformed 
        features of each sample.
    coord_to_label : dict
        Dictionary mapping coordinates (x, y) to a phase/cluster label. Points 
        not in the dictionary are assigned label -1.
    loc_roi : array-like
        Array of coordinates of shape (n_samples, 2), corresponding to the 
        rows in `pca_scores`.
    
    Returns
    -------
    anomalies : np.ndarray or None
        PCA scores of the detected anomalies. Shape (n_anomalies, n_components).
        Returns None if no anomalies are detected.
    anomalies_coords : np.ndarray or None
        Coordinates (x, y) of the detected anomalies. Shape (n_anomalies, 2).
        Returns None if no anomalies are detected.
    anomalies_coords_to_labels : dict or None
        Dictionary mapping each anomaly coordinate (x, y) to its phase/cluster label.
        Returns None if no anomalies are detected.
    """
    loc_roi = np.asarray(loc_roi)
    labels = np.array([coord_to_label.get((x, y), -1) for x, y in loc_roi])
    n_dim = pca_scores.shape[1]
    anomalies = []
    anomalies_coords = []
    anomalies_coords_to_labels = {}
    
    for phase in np.unique(labels):
        mask = (labels == phase)
        if mask.sum() < 2: continue
        
        # Mahalanobis Distance
        data = pca_scores[mask]
        coords = loc_roi[mask]
        cov = np.cov(data.T)
        mean = np.mean(data, axis=0)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        
        diff = data - mean
        distances = np.sum(diff @ inv_cov * diff, axis=1)
        threshold = chi2.ppf(0.95, n_dim)  # 95% confidence interval
        
        phase_anomaly_mask = distances > threshold
        # append the pca scores of the anomalies
        anomalies.append(data[phase_anomaly_mask])
        anomalies_coords.append(coords[phase_anomaly_mask])
        
        # Map coordinates to labels
        for coord in coords[phase_anomaly_mask]:
            anomalies_coords_to_labels[tuple(coord)] = phase
    if anomalies:
        anomalies = np.vstack(anomalies)
        anomalies_coords = np.vstack(anomalies_coords)
        
    else:
        anomalies = None
        anomalies_coords = None
        anomalies_coords_to_labels = None
    return anomalies, anomalies_coords, anomalies_coords_to_labels


# denote the anomalies (box)
def _add_boxes(loc_roi, coords, ax, color, linewidth):
    # find the relative coordinates of reference or anomalies within roi
    # reshape loc_roi
    loc_roi_reshaped = loc_roi.reshape((31, 31, 2))

    # map the coordinates list to local positions
    coord_to_index = {}
    for i in range(31):
        for j in range(31):
            coord = tuple(loc_roi_reshaped[i, j])
            coord_to_index[coord] = (i, j)
    # search the positions of interested anomalies/ reference within roi
    positions = []
    for each in coords:
        key = tuple(each)
        if key in coord_to_index:
            positions.append(coord_to_index[key])
        else:
            positions.append(None) 

    if positions is not None:
        for (x, y) in positions:
            row = int(y)
            col = int(x)
            rect = Rectangle(
                (col-0.5, row-0.5), 1, 1, 
                linewidth=linewidth, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
    
    
def plot_weight_map_pca(pca_scores, loc_roi, anomalies_coords=None, ref1_pos=None, ref2_pos=None, component=0):
    """Plot the weight map with locations of references and anomalies"""
    loc_roi = np.asarray(loc_roi)
    weight_map = np.reshape(pca_scores, (31, 31, 2))
    
    # Obtain the specific component of weight
    data = np.transpose(weight_map[:, :, component])
    
    # colormap
    colors = ["#2ca02c", "#ffffff", "#d62728"]  # green-white-red
    
    abs_max = np.max(np.abs(pca_scores))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    cmap_custom = LinearSegmentedColormap.from_list("custom_diverging", colors)
    
    
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    im = ax.imshow(data, cmap=cmap_custom, norm=norm, interpolation='nearest')
    legend_elements = []
    # denote the anomalies
    if anomalies_coords is not None and len(anomalies_coords) > 0:
        legend_elements.append(
            Line2D([0], [0], color='black', lw=2, label='Anomalies')
        )
        _add_boxes(loc_roi, anomalies_coords, ax, 'black', 2)
    
    # the reference points
    if ref1_pos is not None and len(ref1_pos) > 0:
        legend_elements.append(
            Line2D([0], [0], color='red', lw=3, label='Reference 1')
        )
        _add_boxes(loc_roi, ref1_pos, ax,  'red', 3)
    
    if ref2_pos is not None and len(ref2_pos) > 0:
        legend_elements.append(
            Line2D([0], [0], color='blue', lw=3, label='Reference 2')
        )
        _add_boxes(loc_roi, ref2_pos, ax, 'blue', 3)  
        
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower center',  bbox_to_anchor=(0.5, -0.1), ncol=3)
    # color bar setting
    cbar = plt.colorbar(im)
    cbar.set_ticks([-abs_max, 0, abs_max])
    cbar.ax.set_yticklabels([
        f'pca score = {-abs_max}\n(green)', 
        'pca score = 0\n(white)', 
        f'pca score = {abs_max}\n(red)'
    ], fontsize=10)

    
    
    plt.title(f"Component {component+1} Weight Map with Annotations")
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def reconstruct_pca_signals(
    file_list,
    loc_array,                 # (N, 2) integer coords aligned with file_list
    pca_scores,                # (N, K)
    pca,                       # fitted sklearn PCA (has mean_ and components_)
    output_dir,
    pattern_height=400,
    pattern_width=400,
    slice_x=(0, 100),
    slice_y=(0, 100),
    compare_coords=None        # e.g. [(10,12), (25,3)] — only these cached for plotting
):
    """
    Reconstruct ROI images from a trained PCA model, save reconstructions, and
    return residual stats. Optionally cache selected coordinates for later plotting.

    Notes on ROI:
      - `slice_x`, `slice_y` are **half-open** index ranges [start, end), like NumPy slicing.
        Height = slice_y[1] - slice_y[0], width = slice_x[1] - slice_x[0].
      - `signal_process(file, flag="ROI", pattern_height, pattern_width, slice_x, slice_y)`
        must return the ROI as a flat vector of length height*width.

    Returns:
      residuals        : list[np.ndarray] of shape (H, W)
      residual_norms   : np.ndarray, shape (N,)
      rmses            : np.ndarray, shape (N,)
      ssims            : np.ndarray, shape (N,)
      compare_cache    : list[dict] with keys:
                         {'xy','orig','recon','res','rmse','ssim','name'}
                         (only for coords in `compare_coords`)
    """
    os.makedirs(output_dir, exist_ok=True)
    recon_dir = os.path.join(output_dir, "reconstructed")
    os.makedirs(recon_dir, exist_ok=True)

    # map (x,y) -> index; ensure ints to match user-provided tuples
    loc_int = np.asarray(loc_array, dtype=int)
    coord_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(loc_int)}

    # normalize compare list into a set of coords present in loc_array
    compare_set = set()
    if compare_coords:
        for xy in compare_coords:
            xy_int = (int(xy[0]), int(xy[1]))
            if xy_int in coord_to_idx:
                compare_set.add(xy_int)
            else:
                print(f"[warn] compare coord {xy_int} not found in loc_array; skipping.")

    # ROI size (half-open ranges)
    height = int(slice_y[1] - slice_y[0])
    width  = int(slice_x[1] - slice_x[0])

    residuals = []
    residual_norms = np.zeros(len(file_list), dtype=float)
    rmses = np.zeros(len(file_list), dtype=float)
    ssims = np.zeros(len(file_list), dtype=float)
    compare_cache = []

    for i, file_path in enumerate(file_list):
        # ---- load & preprocess original ROI ----
        orig_vector = signal_process(
            file_path,
            flag="ROI",
            pattern_height=pattern_height,
            pattern_width=pattern_width,
            slice_x=slice_x,
            slice_y=slice_y
        ).flatten()
        orig_img = orig_vector.reshape(height, width)

        # ---- PCA reconstruction ----
        recon_vector = pca.mean_ + np.dot(pca_scores[i], pca.components_)
        recon_img = recon_vector.reshape(height, width)

        # ---- residual + metrics ----
        res = orig_img - recon_img
        residuals.append(res)

        rn = float(np.linalg.norm(res))
        residual_norms[i] = rn
        rmses[i] = rn / np.sqrt(height * width)

        omin, omax = float(orig_img.min()), float(orig_img.max())
        dr = max(1e-8, omax - omin)
        o01 = np.clip((orig_img - omin) / dr, 0, 1)
        r01 = np.clip((recon_img - omin) / dr, 0, 1)
        ssims[i] = ssim(o01, r01, data_range=1.0)

        # ---- save reconstructed as 8-bit grayscale ----
        rmin, rmax = recon_img.min(), recon_img.max()
        if rmax > rmin:
            recon_norm01 = (recon_img - rmin) / (rmax - rmin)
        else:
            recon_norm01 = np.zeros_like(recon_img)
        img_8bit = (np.clip(recon_norm01, 0, 1) * 255).astype(np.uint8)

        file_name = os.path.basename(file_path)
        recon_name = f"reconstructed_{file_name}"
        cv2.imwrite(os.path.join(recon_dir, recon_name), img_8bit)

        # ---- cache comparison triplet if requested ----
        xy = (int(loc_int[i, 0]), int(loc_int[i, 1]))
        if xy in compare_set:
            compare_cache.append({
                'xy': xy,
                'orig': orig_img,
                'recon': recon_img,
                'res': res,
                'rmse': float(rmses[i]),
                'ssim': float(ssims[i]),
                'name': file_name
            })

    print(f"Done. Reconstructed images saved to: {recon_dir}")
    return residuals, residual_norms, rmses, ssims, compare_cache


def plot_pca_comparisons(compare_cache):
    """
    Show side-by-side (Original / Reconstructed / Residual) for the items
    cached by `reconstruct_pca_signals` (only those coords you requested).

    Args:
      compare_cache: list of dicts with keys:
        {'xy','orig','recon','res','rmse','ssim','name'}
    """
    if not compare_cache:
        print("[info] No comparisons to show (compare_cache is empty).")
        return

    for item in compare_cache:
        orig_img = item['orig']
        recon_img = item['recon']
        res_img = item['res']
        xy = item['xy']

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(
            axes,
            [orig_img, recon_img, res_img],
            ["Original", "Reconstructed", "Residual"]
        ):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        fig.suptitle(
            f"(x={xy[0]}, y={xy[1]})  RMSE={item['rmse']:.4f}  SSIM={item['ssim']:.4f}"
        )
        plt.tight_layout()
        plt.show()