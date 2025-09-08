import numpy as np
import torch
from orix import io
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_prepare import filter_files_by_coordinates, KikuchiDataset, coord_xmap_dict, coord_phase_dict_from_dataframe
from vae_cnn import VAE, train_vae
from reconstruct_visualization import reconstruct_and_visualize, latent_space_visualize, visualize_latent_maps, get_latent_features
import os
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
if __name__ == "__main__":
    # select the Kikuchi Patterns within ROI
    selected_files = filter_files_by_coordinates(
        folder_path="/home/users/zhangqn8/storage/Partially reduced oxides 20 minutes Arbeitsbereich 3 Elementverteilungsdaten 5/Images_Valid/",
        x_range=(250,300),
        y_range=(30,70)
    )
    print(f"{len(selected_files)} Figures are selected within the ROI")
    slice_x = (236,436)
    slice_y = (206,306)
    dataset = KikuchiDataset(selected_files, transform=None, step=0.0263, slice_x=slice_x, slice_y=slice_y)
    print(f"The size of the dataset: {len(dataset)}")
    
    coord, image= dataset.__getitem__(1)
    print(np.shape(image))
    df = pd.read_csv("../micro_segmentation/ebsd_processed_with_grain_boundary.csv")
    coord_phase_dict = coord_phase_dict_from_dataframe(df)
    
    
    # Hyperparameter setting
    latent_dim = 4
    batch_size = 16
    epochs = 50
    learning_rate = 1e-3
    
    # device options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # --- Model instance ---
    model = VAE(latent_dim=latent_dim, input_size=(100,200)).to(device)
    print("--- Model Architecture (LeakyReLU + Tanh Output) ---")
    print(model)
    print("----------------------------------------------------")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data preparation
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    print(f"Data loaded: {len(dataset)} images, Batch size: {batch_size}")
    print("IMPORTANT: Input data assumed to be normalized to [-1, 1] for Tanh output.")
    
    
    train_vae(dataloader, model, device, optimizer, epochs)
    
    # kikuchi Pattern reconstruction and display part of them
    # reconstruct_and_visualize(model, device, dataloader, 5)
    
    
    latents, all_labels, all_x_indices, all_y_indices =get_latent_features(model, dataloader, device, coord_phase_dict, 6600)
    
    # Latent space decomposition: select n samples for visualization
    # print("\n--- Visualizing Latent Space (using cNMF) ---")
    # latent_space_visualize(model, dataloader, device, phase_dict, method='cnmf', max_points=961, components_coords=[(40, 27), (44, 12)]) # Increase batches for better viz
    # Example for 2D latent vectors:
    # df = pd.DataFrame(latents)
    # df.to_csv("latents_output_16feature.csv", index=False)
    fig, axes = visualize_latent_maps(
    latents,
    latent_dim,
    highlight_idx=None,
    cmap='plasma',
    save_idx=[0,1,2,3],
    save_dir='maps',
    save_prefix='dim'
    )