#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main function to initiate cNMF calculation using the constrainedmf
package from here

https://github.com/NSLS-II/constrained-matrix-factorization

please follow installation instructions from there.

Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""

from data_processing import signal_process, get_eds_average, get_region_element_averages
import torch
import numpy as np
import pandas as pd
import os
import cv2
from constrainedmf.nmf.models import NMF
from progress.bar import Bar
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
torch.manual_seed(42)
np.random.seed(42)

def constrained_nmf(X, components):
    input_H = [
        torch.tensor(component[None, :], dtype=torch.float) for component in components
    ]
    n = len(components)
    #print(X.shape)
    #X.shape=(1,pixel width*height),W.shape=(1,2),H.shape(2,pixel width*height)
    nmf = NMF(
        X.shape,
        n_components=n,
        initial_components=input_H,
        fix_components=[True for _ in range(len(input_H))],
    )
    nmf.fit(torch.tensor(X), beta=2)#beta=2 for Euclidean distance; learning curve of loss over timesteps
    return nmf

def normalize_sum(lst):
    """
    return normalization values based on percentages

    Args:
        lst (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not lst:
        return []
    total = sum(lst)
    if total == 0:
        return [0.0 for _ in lst]
    return [x / total for x in lst]

def run_cNMF(ROI, components,height, width, slice_x, slice_y):
    file_list = ROI #path for EBSPs
    weights = []
    mse=[]
    r_square=[]
    bar = Bar("Processing", max=len(file_list)) # initialize a progress bar with a total of "filelist" length iterations
    H = torch.tensor(components, dtype=torch.float) 
    for file in file_list:
        bar.next()

        input_X = signal_process(file, flag="ROI",pattern_height=height, pattern_width=width, slice_x=slice_x, slice_y=slice_y)  #ROI width*height EBSP with feature dimension of pixel width*height; But here is input_X 1*pixel width*height
        
        OUTPUT = constrained_nmf(input_X, components)
        
        learned_weights = OUTPUT.W.detach().numpy()
        
        X=torch.tensor(input_X,dtype=torch.float)
        X_reconstructed = OUTPUT.reconstruct(H, OUTPUT.W)
        
        weights.append(learned_weights)
        mse.append(Metrics(X_reconstructed,X)[0].detach().numpy())
        r_square.append(Metrics(X_reconstructed,X)[1].detach().numpy())
        
    

    bar.finish()
    weights = np.array(weights).squeeze(axis=1) 
    return weights,mse,r_square

def run_cNMF_with_eds(ROI, components_combined, df_element, loc, height, width, slice_x, slice_y,
                      *, x_str='x', y_str= 'y'):
    """
    Run constrained NMF on mixed EBSP and EDS data.
    
    Args:
        ROI: List of file paths for EBSPs
        components_combined: Number of components for NMF
        df_element: DataFrame with element percentages and coordinates
        loc: Array of coordinates corresponding to the ROI files
        height: Pattern height for signal processing
        width: Pattern width for signal processing
        slice_x: Tuple (start, end) for x-axis slicing
        slice_y: Tuple (start, end) for y-axis slicing
        
    Returns:
        tuple: (weights, mse, r_square)
    """
    file_list = ROI
    weights = []
    mse = []
    r_square = []
    
    bar = Bar("Processing", max=len(file_list))
    H = torch.tensor(components_combined, dtype=torch.float)
    
    for i, file in enumerate(file_list):
        bar.next()
        
        # Process EBSP signal
        input_X = signal_process(
            file, 
            flag="ROI", 
            pattern_height=height, 
            pattern_width=width, 
            slice_x=slice_x, 
            slice_y=slice_y
        )
        
        # Get element data for this location from the provided loc array
        x_loc, y_loc = loc[i]
        
        # Find element data for this coordinate
        element_row = df_element[
            (df_element[x_str] == x_loc) & 
            (df_element[y_str] == y_loc)
        ]
        
        if len(element_row) == 0:
            # If no exact match, use nearest neighbor or average
            print(f"Warning: No element data found for ({x_loc}, {y_loc}). Using region average.")
            eds_values = get_region_element_averages(
                df_element, 
                (x_loc-1, x_loc+1), 
                (y_loc-1, y_loc+1),
                x_str,
                y_str
            )
        else:
            # Extract element values and normalize
            element_columns = [col for col in df_element.columns if col not in [x_str, y_str]]
            eds_values = element_row[element_columns].values[0]
            eds_values = eds_values / np.sum(eds_values)  # Normalize to sum=1
        
        # Combine EBSP and EDS data
        eds_values = eds_values.reshape(1, -1)
        input_X = np.hstack((input_X, eds_values))
        
        # Run constrained NMF
        OUTPUT = constrained_nmf(input_X, components_combined)
        learned_weights = OUTPUT.W.detach().numpy()
        
        # Calculate reconstruction metrics
        X = torch.tensor(input_X, dtype=torch.float)
        X_reconstructed = OUTPUT.reconstruct(H, OUTPUT.W)
        
        weights.append(learned_weights)
        mse.append(Metrics(X_reconstructed, X)[0].detach().numpy())
        r_square.append(Metrics(X_reconstructed, X)[1].detach().numpy())
    
    bar.finish()
    weights = np.array(weights).squeeze(axis=1)
    return weights, mse, r_square


def run_cNMF_mixeds(ROI, components_combined, loc, edax):
    file_list = ROI #path for EBSPs
    weights = []
    mse=[]
    r_square=[]
    i = 0
    bar = Bar("Processing", max=len(file_list)) # initialize a progress bar with a total of "filelist" length iterations
    H = torch.tensor(components_combined, dtype=torch.float) 
    for file in file_list:
        bar.next()

        input_X = signal_process(file, flag="ROI")  
        # print(np.shape(input_X))
        x_loc = loc[i][0]
        y_loc = loc[i][1]
        eds_values = get_eds_average(x_loc, y_loc, edax)
        # print(eds_values)
        # normalization
        eds_values_nor = normalize_sum(eds_values)
        # print(eds_values_nor)
        # print(np.shape(eds_values))
        eds_values = np.array(eds_values_nor).reshape(1,-1)
        # print(eds_values)
        i+=1
        input_X = np.hstack((input_X, eds_values))
        # print(input_X)
        OUTPUT = constrained_nmf(input_X, components_combined)
        
        learned_weights = OUTPUT.W.detach().numpy()
        
        X=torch.tensor(input_X,dtype=torch.float)
        X_reconstructed = OUTPUT.reconstruct(H, OUTPUT.W)
        
        weights.append(learned_weights)
        mse.append(Metrics(X_reconstructed,X)[0].detach().numpy())
        r_square.append(Metrics(X_reconstructed,X)[1].detach().numpy())
        
    bar.finish()
    weights = np.array(weights).squeeze(axis=1) 
    return weights,mse,r_square


def Metrics(X_reconstructed, input_X):
    
    #mse
    mse = torch.mean(X_reconstructed - input_X) ** 2
    
    #R_square
    ss_total = torch.sum((input_X - torch.mean(input_X)) ** 2)
    ss_residual = torch.sum((input_X - X_reconstructed) ** 2)
    r_square = 1 - (ss_residual / ss_total)   

    return mse,r_square


# def _plot_cnmf(weights, coord_dict, loc_roi):
    
    weights = np.array(weights).squeeze(axis=1) 
    loc_roi = np.asarray(loc_roi)
    assert loc_roi.shape[0] == weights.shape[0], "make sure the number of samples/rows of pca scores the same with loc"
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
    scatter = plt.scatter(weights[:, 0], weights[:, 1], c=roi_labels, cmap='Set1', alpha=0.7, edgecolors='k')
    
    unique_ids = sorted(set(roi_labels.tolist()))
    
    # add legends
    handles = []
    for pid in unique_ids:
        if pid in name_map:
            color = scatter.cmap(scatter.norm(pid))
            patch = mpatches.Patch(color=color, label=name_map[pid])
            handles.append(patch)
    plt.legend(handles=handles, title='Phase')
    
    plt.xlabel("cNMF Component 1")
    plt.ylabel("cNMF Component 2")
    plt.title("cNMF of EBSD Kikuchi Patterns by Phase index")
    plt.show()
    
def _plot_reference(pos_list, loc_roi, ax, scores, marker, label):
        if pos_list is not None:
            # change the coordinates to the sample index
            idx = [np.where((loc_roi == pos).all(axis=1))[0][0] for pos in pos_list]
            ax.scatter(
                scores[idx, 0], scores[idx, 1],
                s=200, marker=marker, edgecolor='black', 
                facecolor='none', linewidths=2, label=label
            )
        return np.column_stack((scores[idx, 0], scores[idx, 1]))
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
        
def _plot_cnmf(weights, coord_dict, loc_roi, ref1_pos=None, ref2_pos=None, anomalies=None, ellipse_alpha=0.3):
    """
    cnmf scatter map: visualization enhancement

    Args:
        weights: trained by the ranking of roi coordinates (from the top to the bottom, from the left to the right)
        coord_dict : dictionary of coordinates as key and phase index as values
        loc_roi : coordinates of roi
        ref1_pos : coordinates of reference component 1
        ref2_pos : coordinates of reference component 2
        anomalies : coordinates of anomalies
        
    """
    loc_roi = np.asarray(loc_roi)
    assert loc_roi.shape[0] == weights.shape[0], "make sure the number of samples/rows of pca scores the same with loc"
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

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = []
    for l in roi_labels:
        if l in phase_colors:
            colors.append(phase_colors[l])
        else:
            colors.append(default_color)
    # plot the main scatter
    main_scatter = ax.scatter(
        weights[:, 0], weights[:, 1], 
        c=colors, alpha=0.7, edgecolors='k', label='Samples'
    )
    # mark the reference point
    weight_ref1 = _plot_reference(ref1_pos, loc_roi, ax, weights, '*', 'Reference 1')
    weight_ref2 = _plot_reference(ref2_pos, loc_roi, ax, weights, 'P', 'Reference 2')
    print(f"weights for reference 1 are: \n {weight_ref1}")
    print(f"weights for reference 2 are: \n {weight_ref2}")
    # plot the confidence eclipse
    for phase, color in phase_colors.items():
        mask = (roi_labels == phase)
        if mask.sum() > 1: 
            _add_confidence_ellipse(
                ax, weights[mask], color, ellipse_alpha
            )
    
    # anomalies plotting
    if anomalies is not None:
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
        
    plt.xlabel("cNMF Component 1")
    plt.ylabel("cNMF Component 2")
    plt.title("cNMF of EBSD Kikuchi Patterns by Phase with Annotations ")
    plt.show()
    
    
# detect the anomalies out of the confidence eclipse
def detect_anomalies_cnmf(weights, coord_dict, loc_roi):
    """
    detect the anomalies out of the confidence eclipse and return the coordinates
    """
    loc_roi = np.asarray(loc_roi)
    labels = np.array([coord_dict.get((x, y), -1) for x, y in loc_roi])
    n_dim = weights.shape[1]
    anomalies = []
    anomalies_coords = []

    for phase in np.unique(labels):
        mask = (labels == phase)
        if mask.sum() < 2: continue
        
        # Mahalanobis Distance
        data = weights[mask]
        cov = np.cov(data.T)
        mean = np.mean(data, axis=0)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        
        diff = data - mean
        distances = np.sum(diff @ inv_cov * diff, axis=1)
        threshold = chi2.ppf(0.95, n_dim)  # 95% confidence interval
        
        # append the pca scores of the anomalies
        phase_anomalies = data[distances > threshold]
        anomalies.append(phase_anomalies)
        # obtain the anomaly coordinates
        phase_indices = np.where(mask)[0]
        anomaly_indices = phase_indices[distances > threshold]
        
        anomalies_coords.extend(loc_roi[anomaly_indices].tolist())
    
    return np.vstack(anomalies) if anomalies else None, np.array(anomalies_coords) if anomalies_coords else None

# denote the anomalies (box)
def _add_boxes(loc_roi, roi_height, roi_width, coords, ax, color, linewidth, hatch=None, alpha=1.0):
    # find the relative coordinates of reference or anomalies within roi
    # reshape loc_roi
    loc_roi_reshaped = loc_roi.reshape((roi_height, roi_width, 2))
    coord_to_index = {}
    # map the coordinates list to local positions
    for j in range(roi_height):
        for i in range(roi_width):
            coord = tuple(loc_roi_reshaped[j, i])  # Note: j,i not i,j
            coord_to_index[coord] = (i, j)  # Store as (col, row)
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
                facecolor='none',
                hatch = hatch,
                alpha =alpha
            )
            ax.add_patch(rect)
    
    
def plot_weight_map_cnmf(weights, loc_roi, anomalies_coords=None, ref1_pos=None, ref2_pos=None, component=0, boundary_locs=None):
    """Plot the weight map with locations of references and anomalies"""
    loc_roi = np.asarray(loc_roi)
    # weights = np.array(weights).squeeze(axis=1) 
    weight_map = np.reshape(weights, (31, 31, 2))
    
    # Obtain the specific component of weight
    data = np.transpose(weight_map[:, :, component])
    
    # colormap
    colors = ["#2ca02c", "#ffffff", "#d62728"]  # green-white-red
    
    
    abs_max = np.max(np.abs(weights))
    # mid point=0.5
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=abs_max)
    # norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

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
    if boundary_locs is not None and len(boundary_locs) > 0:
        legend_elements.append(
            Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Boundary Points')
        )
        _add_boxes(loc_roi, boundary_locs, ax, 'black', 2)    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower center',  bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # color bar setting
    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 0.5, abs_max])
    cbar.ax.set_yticklabels([
        f'Component {component+1} = 0\n(green)', 
        'Both= 0.5\n(white)', 
        f'Component {2 if component==0 else 1} = {abs_max}\n(red)'
    ], fontsize=10)

    plt.title(f"Component {component+1} Weight Map with Annotations")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
def plot_weight_map_cnmf_with_anomalies(weights, loc_roi, roi_height, roi_width, anomalies_dict=None, ref_pos_list=None,   # list of arrays/lists; each inner is shape (M_i, 2) coords
                                    component=0, boundary_locs=None):
    """
    Plot a single cNMF component weight map (2D), with overlays:
      - anomalies (colored by label),
      - any number of reference sets (ref_pos_list),
      - boundary points.

    Notes
    -----
    * The function assumes `weights` are ordered consistently with `loc_roi`,
      and that `loc_roi` covers a rectangular grid of shape (roi_height, roi_width)
      using the same order you used to build `weights`.
    * Overlays are drawn as small square boxes at the nearest grid cells,
      using `_add_boxes(loc_roi, coords_to_mark, ax, color, linewidth, ...)`.
      Make sure `_add_boxes` is available in your namespace.

    Args
    ----
    weights : np.ndarray
        (N, K) cNMF weights for N samples over K components.
    loc_roi : np.ndarray
        (N, 2) integer coordinates for each sample.
    roi_height, roi_width : int
        The ROI grid shape (rows, cols). Must satisfy N == roi_height * roi_width.
    anomalies_dict : dict or None
        Mapping {(x, y): label, ...}. All points appearing in the dict are drawn;
        labels are mapped to distinct colors.
    ref_pos_list : list or None
        List of reference sets. Each element is an array-like of shape (M_i, 2)
        giving (x,y) coords to highlight. Each set is given a distinct marker/color.
        Example: [ref1, ref2, ref3], where ref1/ref2/ref3 are arrays of (x,y).
    component : int
        Which component index to visualize (0-based).
    boundary_locs : array-like or None
        A list/array of (x, y) coords for boundary points to overlay.

    Returns
    -------
    jaccard_index : float or None
    overlap_coefficient : float or None
        Similarity metrics between `anomalies_dict` keys and `boundary_locs`.
        Returned only if both are provided; otherwise None, None.
    """
     # ---- basic checks ----
    weights = np.asarray(weights, float)
    loc_roi = np.asarray(loc_roi)
    N, K = weights.shape
    if N != roi_height * roi_width:
        raise ValueError(f"weights rows ({N}) must equal roi_height*roi_width ({roi_height*roi_width}).")
    if component < 0 or component >= K:
        raise ValueError(f"component={component} is out of range [0, {K-1}].")
    if loc_roi.shape != (N, 2):
        raise ValueError("loc_roi must be (N, 2).")
    
    # weight_map_3d: (roi_height, roi_width, K)
    weight_map = weights.reshape(roi_height, roi_width, K)
    
    # Obtain the specific component of weight
    data = weight_map[:, :, component]
    
    # Create colormap (green-white-red)
    colors = ["#2ca02c", "#ffffff", "#d62728"]
    
    cmap_custom = LinearSegmentedColormap.from_list("custom_diverging", colors)
    
    # Safer vmax: the 95th percentile of the chosen component (avoid outliers)
    comp_vals = data.ravel()
    vmax = float(np.nanpercentile(comp_vals, 95))
    vmax = max(vmax, 1e-6)  # avoid zero
    # With this:
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=vmax)

    
   # ---- plot base map ----
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap=cmap_custom, norm=norm, interpolation='nearest', origin='upper')

    legend_elements = []
    
    # === Add anomalies points with different colors per label ===
    if anomalies_dict is not None and len(anomalies_dict) > 0:
        unique_labels = sorted(set(anomalies_dict.values()))
        color_map = plt.cm.get_cmap('tab10', len(unique_labels))
        for label in unique_labels:
            label_coords = [coord for coord, lbl in anomalies_dict.items() if lbl == label]
            color = color_map(label)  # Use label for color index
            _add_boxes(loc_roi, roi_height, roi_width, label_coords, ax, color, 2, hatch='////', alpha=0.8)
    
    
    # ---- references: any number of sets ----
    # give each set a distinct marker/color
    if ref_pos_list:
        ref_markers = ['*', 'P', 'X', 'D', '^', 's', 'o', 'v', '<', '>']
        ref_colors  = plt.cm.Set2(np.linspace(0, 1, min(10, len(ref_pos_list))))
        for idx, ref_set in enumerate(ref_pos_list):
            if ref_set is None or len(ref_set) == 0:
                continue
            ref_arr = np.asarray(ref_set)
            if ref_arr.ndim != 2 or ref_arr.shape[1] != 2:
                raise ValueError(f"ref_pos_list[{idx}] must be array-like of shape (M, 2).")
            # We reuse _add_boxes; the marker style is represented by legend only
            _add_boxes(loc_roi, roi_height, roi_width, ref_arr, ax, ref_colors[idx % len(ref_colors)], linewidth=3)
            legend_elements.append(
                Line2D([0], [0],
                       marker=ref_markers[idx % len(ref_markers)],
                       color=ref_colors[idx % len(ref_colors)],
                       markerfacecolor='none', markeredgecolor=ref_colors[idx % len(ref_colors)],
                       lw=0, label=f'Reference set {idx+1}')
            )
    
    # Add boundary points with black dashed boxes and cross hatch
    if boundary_locs is not None and len(boundary_locs) > 0:
        legend_elements.append(
            Line2D([0], [0], color='black', lw=2, label='Boundary Points')
        )

        _add_boxes(loc_roi,roi_height, roi_width, boundary_locs, ax, 'black', 2)
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # Color bar settings
    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 0.5, vmax])
    cbar.set_ticklabels([
        f'0.0 (green)',
        '0.5 (white)',
        f'{vmax:.2f} (red)'
    ])
    plt.title(f"Component {component+1} Weight Map with Annotations")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Calculate similarity metrics if both anomalies and boundary points are provided
    jaccard_index = None
    overlap_coefficient = None
    
    if anomalies_dict is not None and boundary_locs is not None:
        # Convert coordinates to sets of tuples (rounded to nearest integer)
        anomalies_set = set(tuple(int(round(v)) for v in coord) for coord in anomalies_dict.keys())
        boundary_set = set(tuple(int(round(v)) for v in point) for point in boundary_locs)
        
        # Calculate intersection and union
        intersection = anomalies_set & boundary_set
        union = anomalies_set | boundary_set
        
        # Calculate Jaccard Index
        if len(union) > 0:
            jaccard_index = len(intersection) / len(union)
        
        # Calculate Overlap Coefficient
        min_size = min(len(anomalies_set), len(boundary_set))
        if min_size > 0:
            overlap_coefficient = len(intersection) / min_size
        
        # Print the similarity metrics
        print(f"Similarity between anomalies and boundary points:")
        print(f"  Number of anomalies: {len(anomalies_set)}")
        print(f"  Number of boundary points: {len(boundary_set)}")
        print(f"  Intersection: {len(intersection)}")
        print(f"  Jaccard Index: {jaccard_index:.4f}")
        print(f"  Overlap Coefficient: {overlap_coefficient:.4f}")
    
    return jaccard_index, overlap_coefficient

def analyze_classify_boundary_and_plot(
    weights: np.ndarray,
    loc: np.ndarray,                                  # (N,2) integer coordinates
    coor_phase_dict: dict,                            # {(x,y): phase_id}  -- “ground truth” for eval
    phase_labels: dict,                               # {phase_id: phase_name}
    cluster_name_map: dict,                           # {cluster_or_label_id: phase_name}
    roi_height: int,
    roi_width: int,
    *,
    component: int = 0,
    anomalies_dict: dict | None = None,               # {(x,y): label}
    ref_pos_list: list | None = None,                 # [array_like(M_i,2), ...]
    coord_to_label: dict | None = None,               # NEW: {(x,y) -> label_id}, overrides argmax(labels)
    figsize=(12, 10),
    normalize: bool = False
):
    """
    One-stop routine:
      1) Build a (x,y)->label_id map for all samples:
         - If coord_to_label is provided, use it (and fall back to argmax only for missing coords).
         - Else, use argmax(weights).
      2) Evaluate against coor_phase_dict on overlapping coordinates.
      3) Detect boundary points (4-neighborhood) using the final label map.
      4) Plot a single component's weight map (heatmap) with overlays; the "white"
         level is centered at the mean of max(weights[row]) over boundary pixels
         (falls back to 0.5 if no boundary).
    """
    # -------- basic checks --------
    weights = np.asarray(weights, float)
    loc = np.asarray(loc)
    N, K = weights.shape
    if loc.shape != (N, 2):
        raise ValueError("loc must have shape (N, 2).")
    if N != roi_height * roi_width:
        raise ValueError(f"N must equal roi_height*roi_width ({roi_height*roi_width}).")
    if not (0 <= component < K):
        raise ValueError(f"component={component} out of range [0,{K-1}].")

    # (x,y) -> row idx
    loc2idx = {tuple(map(int, loc[i])): i for i in range(N)}

    # ---------- normalize per-row (optional) ----------
    if normalize:
        sums = weights.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        weights = weights / sums

    # ---------- labels: coord_to_label overrides argmax ----------
    argmax_labels = np.argmax(weights, axis=1)  # (N,)
    pred_loc2label = {}
    for xy, i in loc2idx.items():
        if coord_to_label is not None and xy in coord_to_label:
            pred_loc2label[xy] = int(coord_to_label[xy])
        else:
            pred_loc2label[xy] = int(argmax_labels[i])

    # ---------- evaluation vs ground truth (overlapping coords) ----------
    common_coords = [xy for xy in pred_loc2label.keys() if xy in coor_phase_dict]
    if len(common_coords) == 0:
        acc = np.nan
        cm_df = pd.DataFrame()
        cls_report = "No overlapping coordinates to evaluate."
    else:
        y_true_names = [str(phase_labels.get(coor_phase_dict[xy], "Unknown")) for xy in common_coords]
        y_pred_names = [str(cluster_name_map.get(pred_loc2label[xy], f"cluster_{pred_loc2label[xy]}"))
                        for xy in common_coords]
        acc = accuracy_score(y_true_names, y_pred_names)
        labels_sorted = sorted(set(y_true_names) | set(y_pred_names))
        cm = confusion_matrix(y_true_names, y_pred_names, labels=labels_sorted)
        cm_df = pd.DataFrame(cm,
                             index=pd.Index(labels_sorted, name="True"),
                             columns=pd.Index(labels_sorted, name="Pred"))
        cls_report = classification_report(y_true_names, y_pred_names,
                                           labels=labels_sorted, zero_division=0)

    # ---------- boundary via 4-neighborhood on *final* labels ----------
    boundary_set = set()
    for (x, y), lab in pred_loc2label.items():
        # 4-neighbors (left, right, up, down)
        for nb in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if nb in pred_loc2label and pred_loc2label[nb] != lab:
                boundary_set.add((x, y))
                break
    # (optional) consistent order for plotting
    boundary_locs = sorted(boundary_set, key=lambda t: (t[1], t[0]))

    # ---------- plotting: component weight heatmap ----------
    weight_map = weights.reshape(roi_height, roi_width, K)
    data = weight_map[:, :, component]

    # center white at mean of max(weights[row]) restricted to boundary pixels
    if len(boundary_locs) > 0:
        max_vals_on_boundary = []
        for xy in boundary_locs:
            idx = loc2idx.get(xy, None)
            if idx is not None:
                max_vals_on_boundary.append(float(np.max(weights[idx])))
        vcenter = float(np.mean(max_vals_on_boundary)) if len(max_vals_on_boundary) > 0 else 0.5
    else:
        vcenter = 0.5

    # green—white—red with white at vcenter
    colors = ["#2ca02c", "#ffffff", "#d62728"]
    cmap_custom = LinearSegmentedColormap.from_list("custom_diverging", colors)

    comp_vals = data.ravel()
    if np.isfinite(comp_vals).any():
        vmax = float(np.nanpercentile(comp_vals, 95))
    else:
        vmax = 1.0
    vmax = max(vcenter + 1e-6, vmax)  # ensure vmax > vcenter
    norm = TwoSlopeNorm(vmin=0.0, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap_custom, norm=norm,
                   interpolation='nearest', origin='upper')

    legend_elements = []

    # anomalies (group by label)
    if anomalies_dict:
        unique_labels = sorted(set(anomalies_dict.values()))
        color_map = plt.cm.get_cmap('tab10', len(unique_labels))
        for idx_c, lab in enumerate(unique_labels):
            coords = [xy for xy, L in anomalies_dict.items() if L == lab]
            _add_boxes(loc, roi_height, roi_width, coords, ax,
                       color=color_map(idx_c), linewidth=2, hatch='////', alpha=0.85)
            legend_elements.append(Line2D([0],[0], color=color_map(idx_c), lw=2, label=f"Anomaly {lab}"))

    # reference sets (any number)
    if ref_pos_list:
        ref_colors = plt.cm.Set2(np.linspace(0, 1, min(10, len(ref_pos_list))))
        ref_markers = ['*', 'P', 'X', 'D', '^', 's', 'o', 'v', '<', '>']
        for i_set, ref_set in enumerate(ref_pos_list):
            if ref_set is None or len(ref_set) == 0:
                continue
            ref_arr = np.asarray(ref_set)
            if ref_arr.ndim != 2 or ref_arr.shape[1] != 2:
                raise ValueError(f"ref_pos_list[{i_set}] must be array-like of shape (M,2).")
            _add_boxes(loc, roi_height, roi_width, ref_arr, ax,
                       color=ref_colors[i_set % len(ref_colors)], linewidth=3, hatch=None, alpha=1.0)
            legend_elements.append(
                Line2D([0],[0], marker=ref_markers[i_set % len(ref_markers)],
                       color=ref_colors[i_set % len(ref_colors)], markerfacecolor='none',
                       markeredgecolor=ref_colors[i_set % len(ref_colors)], lw=0,
                       label=f"Reference {i_set+1}")
            )

    # boundaries (black boxes)
    if len(boundary_locs) > 0:
        _add_boxes(loc, roi_height, roi_width, boundary_locs, ax,
                   color='black', linewidth=2, hatch=None, alpha=1.0)
        legend_elements.append(Line2D([0],[0], color='black', lw=2, label='Boundary'))

    # legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower center',
                  bbox_to_anchor=(0.5, -0.08), ncol=3)

    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, vcenter, vmax])
    cbar.set_ticklabels([f"0.0 (green)",
                         f"{vcenter:.3f} (white @ boundary mean max)",
                         f"{vmax:.3f} (red)"])

    ax.set_title(f"Component {component+1} Weight Map (labels/boundary from coord_to_label if provided)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    metrics = {
        "accuracy": float(acc) if acc == acc else np.nan,
        "confusion_matrix": cm_df,
        "classification_report": cls_report
    }
    # boundary_locs as dict {coord -> label_id}
    boundary_dict = {xy: pred_loc2label[xy] for xy in boundary_locs}
    return {
        "pred_loc2label": pred_loc2label,
        "metrics": metrics,
        "boundary_dict": boundary_dict
    }
    
def reconstruct_weighted_signals(
    file_list,
    loc_array,             # (n_files, 2) -> (x, y)
    weights,               # (n_files, K)
    components,            # (K, H*W) basis patterns
    output_dir,
    height=150,
    width=150,
    compare_coords=None    # e.g. [(10,12), (25,3)]
):
    
    """
    Reconstruct images from CNMF weights and components, save grayscale reconstructions,
    and (optionally) show Original/Reconstructed/Residual for selected coordinates.

    Returns:
        residuals: list of (H,W) residual arrays
        residual_norms: (n_files,) L2 norms
        rmses: (n_files,) RMSE values
        ssims: (n_files,) SSIM values
    """
    os.makedirs(output_dir, exist_ok=True)
    recon_dir = os.path.join(output_dir, "reconstructed")
    os.makedirs(recon_dir, exist_ok=True)

    # --- shape checks / coercion ---
    weights = np.asarray(weights)
    components = np.asarray(components)
    if components.ndim != 2:
        raise ValueError(f"`components` must be (K, H*W), got {components.shape}")
    K, D = components.shape
    if D != height * width:
        raise ValueError(f"`components` second dim must equal H*W ({height*width}), got {D}")
    if weights.ndim != 2 or weights.shape[1] != K:
        raise ValueError(f"`weights` must be (n_files, K={K}), got {weights.shape}")
    if len(file_list) != len(weights) or len(loc_array) != len(weights):
        raise ValueError("file_list, loc_array, and weights must have same length in dim 0")

    loc_int = np.asarray(loc_array, dtype=int)
    compare_set = set()
    if compare_coords:
        compare_set = {(int(x), int(y)) for x, y in compare_coords}

    residuals = []
    residual_norms = np.zeros(len(file_list), dtype=float)
    rmses = np.zeros(len(file_list), dtype=float)
    ssims = np.zeros(len(file_list), dtype=float)

    compare_cache = []

    # Precompute (optional): if components are big, ensure float32 to save memory
    components = components.astype(np.float32, copy=False)

    for i, file_path in enumerate(file_list):
        # ----- load original grayscale (your signal_process already returns ROI) -----
        orig_vector = signal_process(file_path, flag="ROI").flatten().astype(np.float32)
        if orig_vector.size != D:
            raise ValueError(f"signal size {orig_vector.size} != H*W {D}")
        orig_img = orig_vector.reshape(height, width)

        # ----- CNMF reconstruction: weights[i] @ components -----
        # shapes: (K,) @ (K,D) -> (D,)
        recon_vector = np.dot(weights[i].astype(np.float32), components)
        recon_img = recon_vector.reshape(height, width)

        # ----- residual & metrics -----
        residual = orig_img - recon_img
        residuals.append(residual)

        # L2 norm and RMSE
        rn = float(np.linalg.norm(residual))
        residual_norms[i] = rn
        rmses[i] = rn / np.sqrt(height * width)

        # SSIM in 0..1 range using original's dynamic range
        omin, omax = float(orig_img.min()), float(orig_img.max())
        data_range = max(1e-8, omax - omin)
        o01 = np.clip((orig_img - omin) / data_range, 0, 1)
        r01 = np.clip((recon_img - omin) / data_range, 0, 1)
        ssims[i] = ssim(o01, r01, data_range=1.0)

        # ----- save reconstructed as 8-bit grayscale (min-max per image) -----
        rmin, rmax = recon_img.min(), recon_img.max()
        if rmax > rmin:
            recon_norm01 = (recon_img - rmin) / (rmax - rmin)
        else:
            recon_norm01 = np.zeros_like(recon_img)
        img_8bit = (np.clip(recon_norm01, 0, 1) * 255).astype(np.uint8)

        file_name = os.path.basename(file_path)
        out_name = f"reconstructed_{file_name}"
        cv2.imwrite(os.path.join(recon_dir, out_name), img_8bit)

        # queue comparison if requested
        xy = (int(loc_int[i, 0]), int(loc_int[i, 1]))
        if xy in compare_set:
            compare_cache.append({
                'xy': xy,
                'orig': orig_img,
                'recon': recon_img,
                'res': residual,
                'rmse': float(rmses[i]),
                'ssim': float(ssims[i]),
                'name': file_name
            })

    # ----- show comparisons after the loop (grayscale) -----
    for it in compare_cache:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(
            axes,
            [it['orig'], it['recon'], it['res']],
            ["Original", "Reconstructed", "Residual"]
        ):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f"(x={it['xy'][0]}, y={it['xy'][1]})  RMSE={it['rmse']:.4f}  SSIM={it['ssim']:.4f}")
        plt.tight_layout()
        plt.show()

    print(f"Done. CNMF reconstructions saved to: {recon_dir}")
    return residuals, residual_norms, rmses, ssims
    
    