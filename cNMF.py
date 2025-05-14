#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main function to initiate cNMF calculation using the constrainedmf
package from here

https://github.com/NSLS-II/constrained-matrix-factorization

please follow installation instructions from there.

Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""

from data_processing import signal_process, get_eds_average
import torch
import numpy as np
from constrainedmf.nmf.models import NMF
from progress.bar import Bar
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


def constrained_nmf(X, components):
    input_H = [
        torch.tensor(component[None, :], dtype=torch.float) for component in components
    ]
    #print(X.shape)
    #X.shape=(1,pixel width*height),W.shape=(1,2),H.shape(2,pixel width*height)
    nmf = NMF(
        X.shape,
        n_components=2,
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

def run_cNMF(ROI, components):
    file_list = ROI #path for EBSPs
    weights = []
    mse=[]
    r_square=[]
    bar = Bar("Processing", max=len(file_list)) # initialize a progress bar with a total of "filelist" length iterations
    H = torch.tensor(components, dtype=torch.float) 
    for file in file_list:
        bar.next()

        input_X = signal_process(file, flag="ROI")  #ROI width*height EBSP with feature dimension of pixel width*height; But here is input_X 1*pixel width*height
        
        OUTPUT = constrained_nmf(input_X, components)
        
        learned_weights = OUTPUT.W.detach().numpy()
        
        X=torch.tensor(input_X,dtype=torch.float)
        X_reconstructed = OUTPUT.reconstruct(H, OUTPUT.W)
        
        weights.append(learned_weights)
        mse.append(Metrics(X_reconstructed,X)[0].detach().numpy())
        r_square.append(Metrics(X_reconstructed,X)[1].detach().numpy())
        
    

    bar.finish()
    
    return weights,mse,r_square

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
    weights = np.array(weights).squeeze(axis=1) 
    labels = np.array([coord_dict.get((x, y), -1) for x, y in loc_roi])
    
    anomalies = []
    anomalies_coords = []
    for phase in np.unique(labels):
        mask = (labels == phase)
        if mask.sum() < 2: continue
        
        # Mahalanobis Distance
        data = weights[mask]
        cov = np.cov(data.T)
        mean = np.mean(data, axis=0)
        inv_cov = np.linalg.inv(cov)
        
        diff = data - mean
        distances = np.sum(diff @ inv_cov * diff, axis=1)
        threshold = chi2.ppf(0.95, 2)  # 95% confidence interval
        
        # append the pca scores of the anomalies
        phase_anomalies = data[distances > threshold]
        anomalies.append(phase_anomalies)
        # obtain the anomaly coordinates
        phase_indices = np.where(mask)[0]
        anomaly_indices = phase_indices[distances > threshold]
        
        anomalies_coords.extend(loc_roi[anomaly_indices].tolist())
    
    return np.vstack(anomalies) if anomalies else None, np.array(anomalies_coords) if anomalies_coords else None

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
    
    
def plot_weight_map_cnmf(weights, loc_roi, anomalies_coords=None, ref1_pos=None, ref2_pos=None, component=0):
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
    
