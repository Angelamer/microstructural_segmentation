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

class VAEWithCBAM(nn.Module):
    def __init__(self, latent_dim):
        super(VAEWithCBAM, self).__init__()
        self.latent_dim = latent_dim
        
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
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var
    
# --- VAE Loss Function (MSE + KLD) ---
def vae_loss(recon_x, x, mu, log_var, kl_weight=1.0):
    # L2 reconstruction loss
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return mse_loss + kl_weight * kl_loss, mse_loss, kl_loss

# --- VAE with CBAM Training ---
def train_vae_with_cbam(dataloader, model, device, optimizer, num_epochs=20):
    """
    Train the VAE with CBAM attention modules
    
    Args:
        dataloader: DataLoader containing training data
        model: VAEWithCBAM model instance
        device: torch device (cuda/cpu)
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        
        for batch_idx, (batch_indices, data) in enumerate(dataloader):
            # Move data to device and ensure correct dtype
            data = data.to(device).float()  # Convert to float32
            
            # Forward pass
            optimizer.zero_grad()
            recon_data, mu, log_var = model(data)
            
            # Calculate loss
            loss, mse_loss, kl_loss = vae_loss(recon_data, data, mu, log_var)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            batch_size = len(data)
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_kl += kl_loss.item()
            
            # Print batch statistics
            if batch_idx % (batch_size-1) == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {loss.item()/batch_size:.4f}, '
                      f'MSE: {mse_loss.item()/batch_size:.4f}, '
                      f'KLD: {kl_loss.item()/batch_size:.4f}')

        # Print epoch statistics
        avg_loss = total_loss / len(dataloader.dataset)
        avg_mse = total_mse / len(dataloader.dataset)
        avg_kld = total_kl / len(dataloader.dataset)
        print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f} '
              f'(MSE: {avg_mse:.4f}, KLD: {avg_kld:.4f})')

    print("Training finished.")
    
    # Save the model
    torch.save(model.state_dict(), 'vae_cbam_model.pth')
    print("Model saved to vae_cbam_model.pth")

    return model