import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 120x120 -> 60x60
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 60x60 -> 30x30
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 30x30 -> 15x15
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 3, stride=2, padding=1), # 15x15 -> 8x8
            nn.ReLU(),
        )
        # (batch, latent_dim, 8, 8)
        self.flattened_size = latent_dim * 8 * 8
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),         # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),         # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),          # 64x64 -> 128x128
            nn.Tanh(),
        )
    def encode(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, x.size(1)//(8*8), 8, 8)
        return self.decoder(x)
    def center_crop(self, x, target_height, target_width):
        _, _, h, w = x.shape
        start_y = (h - target_height) // 2
        start_x = (w - target_width) // 2
        return x[:, :, start_y:start_y+target_height, start_x:start_x+target_width]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        recon_x = self.center_crop(recon_x, 120, 120)
        
        return recon_x, mu, logvar
# --- VAE Loss Function (MSE + KLD) ---
def vae_loss(recon_x, x, mu, log_var, kl_weight=1.0):
    # L2 reconstruction loss
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return mse_loss + kl_weight * kl_loss, mse_loss, kl_loss

# --- VAE Training ---
def train_vae(dataloader, model, device, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        
        for batch_idx, (batch_indices, data) in enumerate(dataloader):
            data = data.to(device) # Data should be in [-1, 1] range
            
            # Ensure data has the correct dtype (float32 is standard for models)
            if data.dtype != torch.float32:
                data = data.to(dtype=torch.float32)

            optimizer.zero_grad()
            recon_data, mu, log_var = model(data) # recon_batch will be in [-1, 1] range
            
            # Calculate individual loss components for monitoring
            loss, mse_loss, kl_loss = vae_loss(recon_data, data, mu, log_var)
            loss.backward()
            optimizer.step()
            

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_kl += kl_loss.item()
            
            if batch_idx % 30 == 0:   # 30 = max(batch_idx)
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                    f'Loss: {loss.item() / len(data):.4f}, '
                    f'MSE: {mse_loss.item() / len(data):.4f}, '
                    f'KLD: {kl_loss.item() / len(data):.4f}')


        avg_loss = total_loss / len(dataloader.dataset)
        avg_mse = total_mse / len(dataloader.dataset)
        avg_kld = total_kl / len(dataloader.dataset)
        print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, KLD: {avg_kld:.4f})')

    print("Training finished.")
    
    # --- Save the model ---
    # torch.save(model.state_dict(), 'vae_model_leaky_tanh.pth')
    # print("Model saved to vae_model_leaky_tanh.pth")
