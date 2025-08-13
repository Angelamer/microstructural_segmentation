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
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
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




# --- ResNet Basic Block ---
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else None
        )
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.cbam(out)
        out = self.relu(out)
        return out

# --- Encoder with Residual Blocks ---
class ResNetCBAM(nn.Module):
    def __init__(self, latent_dim=32, in_channels=1):
        super(ResNetCBAM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)
        self.layer4 = BasicBlock(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_latent = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)            
        x = self.avgpool(x)            # [B,128,1,1]
        x = torch.flatten(x, 1)        # [B,128]
        z = self.fc_latent(x)          # [B, latent_dim]
        return z

class ResNetCBAMAutoencoder(nn.Module):
    def __init__(self, latent_dim=32, in_channels=1):
        super().__init__()
        self.encoder = ResNetCBAM(latent_dim, in_channels)  # Your encoder from before

        # Simple decoder example: transpose conv layers (you can design deeper/better!)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),              # project up
            nn.ReLU(),
            nn.Linear(128, 16 * 15 * 15),            # match a 15x15x16 feature map (for upsampling)
            nn.ReLU(),
            nn.Unflatten(1, (16, 15, 15)),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # (30x30)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),   # (60x60)
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1),   # (120x120)
            nn.Tanh(),  # if input normalized to [-1,1]
        )
    
    def forward(self, x):
        z = self.encoder(x)   # [B, latent_dim]
        recon = self.decoder(z)
        return z,recon       # Return both reconstruction and latent


def train_resnet_withcbam(model, dataloader, optimizer, device, loss_fn, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (coords, data) in enumerate(dataloader):
            data = data.to(device).float()
            optimizer.zero_grad()
            _, recon= model(data)
            # Ensure recon and data are same shape!
            loss = loss_fn(recon, data)
            loss.backward()
            optimizer.step()
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size

            if batch_idx % 30 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                      f'Reconstruction Loss: {loss.item() / batch_size:.4f}')
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'====> Epoch: {epoch+1} Average Recon Loss: {avg_loss:.4f}')
    print("Training finished.")
    torch.save(model.state_dict(), 'resnet_cbam_autoencoder.pth')
    print("Model saved to resnet_cbam_autoencoder.pth")
    return model