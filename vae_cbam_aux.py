import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class VAEWithCBAM_AUX(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(VAEWithCBAM_AUX, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # ========== Encoder with CBAM ==========
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=9, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(4),  #CBAM
            
            nn.Conv2d(4, 8, kernel_size=9, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(8),  #CBAM
            
            nn.Conv2d(8, 16, kernel_size=9, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(16),  #CBAM
            
            nn.Conv2d(16, 32, kernel_size=8, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(32)  #CBAM
        )
        
        self.fc_mu = nn.Linear(32 * 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(32 * 1 * 1, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 32 * 4 * 4)
        
        # Auxiliary prediction head
        self.auxiliary_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))
        
        # ========== Decoder with CBAM ==========
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=9, stride=2, padding=2, output_padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(16),  #CBAM
            
            nn.ConvTranspose2d(16, 8, kernel_size=9, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(8),  #CBAM
            
            nn.ConvTranspose2d(8, 4, kernel_size=9, stride=2, padding=0, output_padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(4),  #CBAM
            
            nn.ConvTranspose2d(4, 1, kernel_size=9, stride=2, padding=5, output_padding=1),
            nn.Tanh()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 32, 4, 4)
        return self.decoder(x)
    
    def predict_auxiliary(self, z):
        """Predict physical properties from latent space"""
        return self.auxiliary_head(z)
    
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        aux_pred = self.predict_auxiliary(z)
        return recon_x, mu, log_var, aux_pred
    
def vae_aux_loss(recon_x, x, mu, log_var, aux_pred, aux_target, 
                recon_weight=1.0, kl_weight=1.0, aux_weight=0.5):
    """
    Combined VAE and auxiliary loss
    
    Args:
        recon_x: Reconstructed image
        x: Original image
        mu: Latent mean
        log_var: Latent log variance
        aux_pred: Auxiliary predictions
        aux_target: Ground truth physical properties
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence
        aux_weight: Weight for auxiliary loss
    """
    # Reconstruction loss
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Auxiliary loss (cross-entropy for classification)
    aux_loss = F.cross_entropy(aux_pred, aux_target, reduction='sum')
    
    total_loss = (recon_weight * mse_loss + 
                 kl_weight * kl_loss + 
                 aux_weight * aux_loss)
    
    return total_loss, mse_loss, kl_loss, aux_loss

# --- VAE with CBAM Training ---
def train_vae_with_cbam_aux(dataloader, model, device, optimizer,phase_dict, num_epochs=20):
    """
    Train the VAE with CBAM attention modules and auxiliary output modification
    
    Args:
        dataloader: DataLoader containing training data
        model: VAEWithCBAM model instance
        device: torch device (cuda/cpu)
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
    """
    model.train()
    # incase there are labels starting from 1, will cause errors when training
    all_labels = []
    for batch_idx, (batch_indices, _) in enumerate(dataloader):
        x_idx_tensor, y_idx_tensor = batch_indices
        x_list = x_idx_tensor.tolist()
        y_list = y_idx_tensor.tolist()
        batch_labels = [phase_dict.get((x, y), -1) for x, y in zip(x_list, y_list)]
        all_labels.extend(batch_labels)
    
    min_label = min(l for l in all_labels if l != -1)  # Exclude invalid labels
    print(f"Minimum phase label in dataset: {min_label}, normalizing to start from 0")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        total_aux = 0.0
        
        for batch_idx, (batch_indices, data) in enumerate(dataloader):
            # Get data and auxiliary targets
            x_idx_tensor, y_idx_tensor = batch_indices
            data = data.to(device).float()
            
            # Get and normalize phase labels
            x_list = x_idx_tensor.tolist()
            y_list = y_idx_tensor.tolist()
            batch_labels = [phase_dict.get((x, y), -1) for x, y in zip(x_list, y_list)]
            batch_labels = torch.tensor(batch_labels, device=device)
            
            
            # Normalize labels and filter invalid (-1) labels
            valid_mask = batch_labels != -1
            aux_targets = batch_labels[valid_mask] - min_label
            data = data[valid_mask]
            
            if len(data) == 0:
                continue  # Skip batches with no valid labels
            
            # Forward pass
            optimizer.zero_grad()
            recon_data, mu, log_var, aux_pred = model(data)
            
            # Calculate loss
            loss, mse_loss, kl_loss, aux_loss = vae_aux_loss(
                recon_data, data, mu, log_var, aux_pred, aux_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            batch_size = len(data)
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_kl += kl_loss.item()
            total_aux += aux_loss.item()
            
            # Logging
            if batch_idx % 30 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                     f'Loss: {loss.item()/batch_size:.4f}, '
                     f'MSE: {mse_loss.item()/batch_size:.4f}, '
                     f'KLD: {kl_loss.item()/batch_size:.4f}, '
                     f'Aux: {aux_loss.item()/batch_size:.4f}')
        
        # Epoch statistics
        avg_loss = total_loss / len(dataloader.dataset)
        avg_mse = total_mse / len(dataloader.dataset)
        avg_kl = total_kl / len(dataloader.dataset)
        avg_aux = total_aux / len(dataloader.dataset)
        
        print(f'====> Epoch: {epoch+1} Average losses - '
             f'Total: {avg_loss:.4f}, MSE: {avg_mse:.4f}, '
             f'KLD: {avg_kl:.4f}, Aux: {avg_aux:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'vae_cbam_aux.pth')
    print("Model saved to vae_cbam_aux.pth")
    
    return model