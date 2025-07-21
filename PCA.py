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
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle



def run_PCA(ROI,components):
    file_list = ROI
    bar = Bar("Processing", max=len(file_list))
    signal_processed = []
    
    for file in file_list:
        bar.next()
        signal_processed.append(signal_process(file, flag="ROI"))  #31*31 EBSP with feature dimension of 150*150; But here is input_X 1*90000
        
    
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
def _plot_reference(ref_pos, loc_roi, ax, pca_scores, marker, label):
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
        if len(data) < 2: return 
        
        # compute the eclipse parameters
        cov = np.cov(data.T)
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(lambda_)
        chi = np.sqrt(chi2.ppf(0.95, 2))  # 95% confidence interval
        
        # plot the eclipse
        ell = Ellipse(
            xy=np.mean(data, axis=0),
            width=lambda_[0]*chi*2, 
            height=lambda_[1]*chi*2,
            angle=np.degrees(np.arctan2(v[1,0], v[0,0])),
            edgecolor=color, 
            facecolor='none', 
            linestyle='--', 
            alpha=alpha
        )
        ax.add_patch(ell)
    
def _plot_pca(pca_scores, coord_dict, loc_roi, dim=2, ref1_pos=None, ref2_pos=None, anomalies=None, ellipse_alpha=0.3):
    """
    PCA scatter map: visualization enhancement

    Args:
        pca_scores: trained by the ranking of roi coordinates
        coord_dict : dictionary of coordinates as key and phase index as values
        loc_roi : coordinates of roi
        ref1_pos : coordinates of reference component 1
        ref2_pos : coordinates of reference component 2
        anomalies : coordinates of anomalies
        
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
    _plot_reference(ref1_pos, loc_roi, ax, pca_scores, '*', 'Reference 1')
    _plot_reference(ref2_pos, loc_roi, ax, pca_scores, 'P', 'Reference 2')
    
    
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
    detect the anomalies out of the confidence eclipse and return the coordinates/ anomalies scores/ labels
    phase category is obtained from the indexed file (manual cluster/ group)
    """
    loc_roi = np.asarray(loc_roi)
    labels = np.array([coord_to_label.get((x, y), -1) for x, y in loc_roi])
    n_dim = pca_scores.shape[1]
    anomalies = []
    anomalies_coords = []
    anomalies_labels = []
    
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
        
        anomalies_labels.extend([phase] * np.sum(phase_anomaly_mask))
    if anomalies:
        anomalies = np.vstack(anomalies)
        anomalies_coords = np.vstack(anomalies_coords)
        anomalies_labels = np.array(anomalies_labels)
    else:
        anomalies = None
        anomalies_coords = None
        anomalies_labels = None
    return anomalies, anomalies_coords, anomalies_labels


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
    
