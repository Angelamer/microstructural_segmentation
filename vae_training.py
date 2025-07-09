import numpy as np
import torch
from orix import io
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_prepare import filter_files_by_coordinates, KikuchiDataset, coord_xmap_dict
from vae_cnn import VAE, train_vae
from reconstruct_visualization import reconstruct_and_visualize, latent_space_visualize, visualize_latent_maps, get_latent_features
import os
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
if __name__ == "__main__":
    # select the Kikuchi Patterns within ROI
    selected_files = filter_files_by_coordinates(
        folder_path="EBSD_scan",
        x_range=(20,50),
        y_range=(5,35)
    )
    print(f"{len(selected_files)} Figures are selected within the ROI")
    
    dataset = KikuchiDataset(selected_files, transform=None)
    print(f"The size of the dataset: {len(dataset)}")
    
    
    # obtain the index information (commercial software processed results)
    fname_ang = "EBSD_scan/Scan3_cropped.ang"
    xmap = io.load(fname_ang)   


    # construct a dictionary indexed by coordinates to obtain phase_id information (or others/ orientations?)
    coord_index= coord_xmap_dict(xmap)
    
    # Hyperparameter setting
    latent_dim = 16
    batch_size = 32
    epochs = 10
    learning_rate = 1e-4
    
    # device options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # --- Model instance ---
    model = VAE(latent_dim=latent_dim).to(device)
    print("--- Model Architecture (LeakyReLU + Tanh Output) ---")
    print(model)
    print("----------------------------------------------------")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data preparation
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Data loaded: {len(dataset)} images, Batch size: {batch_size}")
    print("IMPORTANT: Input data assumed to be normalized to [-1, 1] for Tanh output.")
    
    
    train_vae(dataloader, model, device, optimizer, epochs)
    
    # kikuchi Pattern reconstruction and display part of them
    # reconstruct_and_visualize(model, device, dataloader, 5)
    
    # obtain the corresponding phase id of each kikuchi pattern
    phase_dict = coord_xmap_dict(xmap, 0.05)
    latents, all_labels, all_x_indices, all_y_indices =get_latent_features(model, dataloader, device, phase_dict, 961)
    
    # Latent space decomposition: select n samples for visualization
    # print("\n--- Visualizing Latent Space (using cNMF) ---")
    # latent_space_visualize(model, dataloader, device, phase_dict, method='cnmf', max_points=961, components_coords=[(40, 27), (44, 12)]) # Increase batches for better viz
    # Example for 2D latent vectors:
    df = pd.DataFrame(latents)
    df.to_csv("latents_output_16feature.csv", index=False)
    """
    fig, axes = visualize_latent_maps(
    latents,
    latent_dim,
    highlight_idx=None,
    cmap='plasma',
    save_idx=[0,1],
    save_dir='maps',
    save_prefix='dim'
    )
    """