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
    return pca_scores


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


