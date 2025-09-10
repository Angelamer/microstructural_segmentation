from data_prepare import KikuchiH5Dataset, coord_phase_dict_from_dataframe
import torch
from torch.utils.data import DataLoader
from vae_cnn import VAE
from reconstruct_visualization import get_latent_features, visualize_latent_maps
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
    latent_dim = 4
    batch_size = 64
    epochs = 40
    learning_rate = 1e-3
    kikuchi_p = "../ebsd_kikuchi/20min_processed_signals.h5"
    H, W = 400,500
    roi_xrange = (250,300)
    roi_yrange = (30,70)
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
    state = torch.load("vae_model_leaky_tanh.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    
    
    # kikuchi Pattern reconstruction and display part of them
    # reconstruct_and_visualize(model, device, dataloader, 5)
    
    
    latents, all_labels, all_x_indices, all_y_indices =get_latent_features(model, dataloader, device, coord_phase_dict, None, False)
    
    # Latent space decomposition: select n samples for visualization
    # print("\n--- Visualizing Latent Space (using cNMF) ---")
    # latent_space_visualize(model, dataloader, device, phase_dict, method='cnmf', max_points=961, components_coords=[(40, 27), (44, 12)]) # Increase batches for better viz
    # Example for 2D latent vectors:
    # df = pd.DataFrame(latents)
    # df.to_csv("latents_output_16feature.csv", index=False)
    (fig, axes), maps, axes_info = visualize_latent_maps(
    latents,
    all_x_indices,
    all_y_indices,
    latent_idx=[0,1,2,3],
    cmap='plasma',
    suptitle="latent feature map",
    save_idx=[0,1,2,3],
    save_dir='maps',
    save_prefix='dim',
    roi_xrange=roi_xrange,
    roi_yrange=roi_yrange
    )