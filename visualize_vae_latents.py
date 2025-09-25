from data_prepare import KikuchiH5Dataset, coord_phase_dict_from_dataframe
import torch
from torch.utils.data import DataLoader
from vae_cnn import VAE
from reconstruct_visualization import get_latent_features, visualize_latent_maps
import pandas as pd
from latent_map_selection import (
    load_feature_csv, load_elements_csv,
    run_latent_vs_pca_cnmf_gmm, run_latent_vs_elements
)
import numpy as np
import os
import pandas as pd
if __name__ == "__main__":
    # select the Kikuchi Patterns within ROI
    # selected_files = filter_files_by_coordinates(
    #     folder_path="/home/users/zhangqn8/storage/Partially reduced oxides 20 minutes Arbeitsbereich 3 Elementverteilungsdaten 5/Images_Valid/",
    #     x_range=(250,300),
    #     y_range=(30,70)
    # )
    # print(f"{len(selected_files)} Figures are selected within the ROI")
    # slice_x = (236,436)
    # slice_y = (206,306)
    # dataset = KikuchiDataset(selected_files, transform=None, step=0.0263, slice_x=slice_x, slice_y=slice_y)
    # print(f"The size of the dataset: {len(dataset)}")
    
    # coord, image= dataset.__getitem__(1)
    # print(np.shape(image))
    df = pd.read_csv("../ebsd_kikuchi/ebsd_processed_with_grain_boundary.csv")
    coord_phase_dict = coord_phase_dict_from_dataframe(df)
    # Hyperparameter setting
    latent_dim = 64
    batch_size = 64
    # epochs = 40
    learning_rate = 1e-3
    kikuchi_p = "/home/users/zhangqn8/storage/20min_processed_signals_fullimage.h5"
    # H, W = 400,500
    # roi_xrange = (250,300)
    # roi_yrange = (30,70)
    ds = KikuchiH5Dataset(kikuchi_p, normalize='minus_one_one')
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,              # works now (map-style dataset)
        num_workers=4,             # tune for your I/O
        pin_memory=True,
        persistent_workers=True
    )
    
    # device options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # --- Model instance ---
    model = VAE(latent_dim=latent_dim).to(device)
    print("--- Model Architecture (LeakyReLU + Tanh Output) ---")
    print(model)
    print("----------------------------------------------------")
    
    # warm-up to init FC
    with torch.no_grad():
        _, warm_batch = next(iter(dataloader))
        warm_batch = warm_batch.to(device, dtype=torch.float32, non_blocking=True)
        _ = model(warm_batch)

    # load weights
    state = torch.load("vae_model_for_full_data_dim_64.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    
    
    # kikuchi Pattern reconstruction and display part of them
    # reconstruct_and_visualize(model, device, dataloader, 5)
    
    
    latents, all_labels, all_x_indices, all_y_indices =get_latent_features(model, dataloader, device, coord_phase_dict, None, False)
    # print(all_x_indices)
    # Latent space decomposition: select n samples for visualization
    # print("\n--- Visualizing Latent Space (using cNMF) ---")
    # latent_space_visualize(model, dataloader, device, phase_dict, method='cnmf', max_points=961, components_coords=[(40, 27), (44, 12)]) # Increase batches for better viz
    # Example for 2D latent vectors:
    # df = pd.DataFrame(latents)
    # df.to_csv("latents_output_16feature.csv", index=False)
    # (fig, axes), maps, axes_info = visualize_latent_maps(
    # latents,
    # all_x_indices,
    # all_y_indices,
    # latent_idx=[0,1,2,3,4,5,6,7],
    # cmap='RdBu',
    # suptitle="latent feature map",
    # save_idx=[0,1,2,3,4,5,6,7],
    # save_dir='maps',
    # save_prefix='fulldata_dim',
    # roi_xrange=None,
    # roi_yrange=None
    # )
    
    xs_all = np.asarray(all_x_indices, dtype=int)
    ys_all = np.asarray(all_y_indices, dtype=int)
    
    pca_coords, pca_feats, pca_cols = load_feature_csv("/home/users/zhangqn8/workflow/ebsd_kikuchi/20min_pca_scores.csv", prefix="PCA_")

    out_dir = f"latent_maps_selected_dim_{latent_dim}/"
    os.makedirs(out_dir, exist_ok=True)

    figs1, rep1 = run_latent_vs_pca_cnmf_gmm(
        out_dir=out_dir,
        latents=latents,
        xs_all=np.asarray(xs_all, dtype=int),
        ys_all=np.asarray(ys_all, dtype=int),
        roi_coords=pca_coords,
        roi_feats=pca_feats,
        kmin=2, kmax=12, criterion="bic", metric="ari", topn=3, seed=0
    )
    print(rep1)
    
    cnmf_coords, cnmf_feats, cnmf_cols = load_feature_csv("/home/users/zhangqn8/workflow/ebsd_kikuchi/20min_cnmf_weights.csv", prefix="cNMF_")
    figs2, rep2 = run_latent_vs_pca_cnmf_gmm(
        out_dir=out_dir,
        latents=latents,
        xs_all=np.asarray(xs_all, dtype=int),
        ys_all=np.asarray(ys_all, dtype=int),
        roi_coords=cnmf_coords,
        roi_feats=cnmf_feats,
        kmin=2, kmax=12, criterion="bic", metric="ari", topn=3, seed=0
    )
    print(rep2)
    
    elem_coords_all, elem_values_all, elem_names = load_elements_csv("/home/users/zhangqn8/workflow/ebsd_kikuchi/20min_element_maps.csv")
    roi_xrange = (200,400)
    roi_yrange = (0,100)
    # (i) SUM mode: normalized per-channel, then (optionally weighted) sum
    figs3, rep3 = run_latent_vs_elements(
        out_dir=out_dir,
        latents=latents,
        xs_all=np.asarray(xs_all, dtype=int),
        ys_all=np.asarray(ys_all, dtype=int),
        elem_coords_all=elem_coords_all,
        elem_values_all=elem_values_all,   # shape (N,E)
        elem_names=elem_names,
        roi_xrange=roi_xrange,
        roi_yrange=roi_yrange,
        synth_mode="sum",
        weights=None,                      # or np.array of length E if you want to emphasize some elements
        topn=3
    )
    print(rep3)
    
    
    # (ii) RGB mode (optional): e.g., Fe->R, Ni->G, O->B (provide indices into elem_names)
    # Example: find indices quickly
    elem_idx = {name:i for i,name in enumerate(elem_names)}
    rgb_ids = [elem_idx["Fe"], elem_idx["Al"], elem_idx["O"]]
    figs4, rep4 = run_latent_vs_elements(
        out_dir=out_dir, latents=latents,
        xs_all=np.asarray(xs_all, dtype=int), ys_all=np.asarray(ys_all, dtype=int),
        elem_coords_all=elem_coords_all, elem_values_all=elem_values_all, elem_names=elem_names,
        roi_xrange=roi_xrange, roi_yrange=roi_yrange,
        synth_mode="rgb", rgb_elements=rgb_ids, topn=3
    )
    print(rep4)

    
    