#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for plotting

Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""

# Utility module for plotting functions
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import re

def _plot_grid(coord_grid, img, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), tight_layout={"pad": 0})
    # this is essential for the data to conform to RVB-EBSD reading of fields.
    ax[0].invert_yaxis()
    ax[0].set_aspect("equal")
    ax[0].scatter(coord_grid[:, 0], coord_grid[:, 1], s=0.02, alpha=0.8)
    ax[0].margins(0)
    ax[0].set_xlabel("pattern index (x spatial dimension)")
    ax[0].set_ylabel("pattern index (y spatial dimension)")

    
    ax[1].imshow(img)
    ax[1].axis("off")
    
    # ax[1].set_title("Corresponding Phase Map for reference")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _plot_ROI(ROI, grid, img, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout={"pad": 0})
    # this is essential for the data to conform to RVB-EBSD reading of fields.
    ax[0].invert_yaxis()
    ax[0].set_aspect("equal")
    ax[0].scatter(grid[:, 0], grid[:, 1], s=0.02, alpha=0.8)
    ax[0].margins(0)
    ax[0].set_xlabel("pattern index (x spatial dimension)")
    ax[0].set_ylabel("pattern index (y spatial dimension)")
    ax[0].set_title("The ROI is shown as RED region")
    #X,Y is the x, y value for each EBSP
    X = [lis[1] for lis in ROI]
    Y = [lis[2] for lis in ROI]
    # X = [list[0] for lis in XY]
    ax[0].scatter(X, Y, s=1, alpha=1.0)

    ax[1].tick_params(top="off", bottom="off")
    ax[1].imshow(img)
    ax[1].axis('off')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _plot_component(ROI, grid, img, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout={"pad": 0})
    # this is essential for the data to conform to RVB-EBSD reading of fields.
    ax[0].invert_yaxis()
    ax[0].set_aspect("equal")
    ax[0].scatter(grid[:, 0], grid[:, 1], s=0.02, alpha=0.8)
    ax[0].margins(0)
    ax[0].set_xlabel("pattern index (x spatial dimension)")
    ax[0].set_ylabel("pattern index (y spatial dimension)")
    ax[0].set_title("The ROI is shown as RED region")
    X = [lis[1] for lis in ROI]
    Y = [lis[2] for lis in ROI]

    ax[0].scatter(X, Y, s=1, alpha=1.0)

    ax[1].tick_params(top="off", bottom="off")
    ax[1].imshow(img)
    ax[1].axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        
        
def _safe_savefig(fig, save_dir=None, filename=None, dpi=300):
    if save_dir is None:
        if filename is not None:
            print(f"[warn] filename='{filename}' ignored because save_dir=None")
        return
    os.makedirs(save_dir, exist_ok=True)
    fname = filename or "figure"
    # clean the filename
    fname = re.sub(r"[\\/:*?\"<>|\s]+", "_", fname).strip("_")
    path = os.path.join(save_dir, f"{fname}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path