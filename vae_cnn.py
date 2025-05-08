import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



    
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # ========== Encoder ==========
        # Input: (batch_size, 1, 120, 120)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=9, stride=2, padding=0),          # (120-9)/2 + 1 = 55.5 -> 56; 1 = input_channel; 4 = output_channel
            nn.LeakyReLU(0.2, inplace = True),
            # Output: (batch_size, 4, 56, 56)
            
            nn.Conv2d(4, 8, kernel_size=9, stride=2, padding=0),  
            nn.LeakyReLU(0.2, inplace = True),
            # Output: (batch_size, 8, 24, 24)
            
            nn.Conv2d(8, 16, kernel_size=9, stride=2, padding=0),   
            nn.LeakyReLU(0.2, inplace = True),
            # Output: (batch_size, 16, 8, 8)
            
            nn.Conv2d(16, 32, kernel_size=8, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace = True)
            # Output: (batch_size, 32, 1, 1)
            
        )
        
        # Flatten and FC layers for mu and logvar
        self.fc_mu = nn.Linear(32 * 1 * 1, latent_dim)  # mean value
        self.fc_logvar = nn.Linear(32 * 1 * 1, latent_dim) # log variance

        # Decoder input
        # FC layer to project latent vector z to the size needed for reshaping
        self.fc_decode = nn.Linear(latent_dim, 32 * 4 * 4)
        
        # Transposed Convolutions
        # Input: (batch_size, 32, 1, 1)
        # ========== Decoder ==========
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=9, stride=2, padding=2, output_padding=0), # (4-1)/2 -2*2 + 9 + 0 = 11
            nn.LeakyReLU(0.2, inplace = True),
            # Output: (batch_size, 16, 11, 11)
            
            nn.ConvTranspose2d(16, 8, kernel_size=9, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(0.2, inplace = True),
            # Output: (batch_size, 8, 27, 27)
            
            nn.ConvTranspose2d(8, 4, kernel_size=9, stride=2, padding=0, output_padding=0),
            nn.LeakyReLU(0.2, inplace = True),
            # Output: (batch_size, 4, 61, 61)
            
            nn.ConvTranspose2d(4, 1, kernel_size=9, stride=2, padding=5, output_padding=1),  
            nn.Tanh()
            # Output: (batch_size, 1, 120, 120)

        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # flatten (batch_size, latent_dim)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var
    
    def decode(self, z):
        # print(f"Decoder input z shape: {z.shape}") # Should be [batch, 16]
        x = self.fc_decode(z)
        x = x.view(-1, 32, 4, 4)
        # print(f"Shape before dec_tconv1 (after view): {x.shape}") # Should be [batch, 32, 4, 4]
        # the start channels, decoder dimensions [32, 4, 4]
        return self.decoder(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # standard deviation
        eps = torch.randn_like(std) # Sample epsilon from N(0, I)
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








