import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


    
class VAE(nn.Module):
    """
    Input: (B,1,H,W). n_down stride-2 convs in encoder; decoder upsamples back
    to the recorded sizes. Fully-connected layers are lazily initialized from
    the first batch so any H,W works (as long as all batches use the same H,W).
    """
    def __init__(self, latent_dim=16, base_channels=32, n_down=5, upsample_mode="nearest", output_mode="minus_one_one"):
        super().__init__()
        self.latent_dim = latent_dim
        self.base = base_channels
        self.n_down = n_down
        self.upsample_mode = upsample_mode  # 'nearest' or 'bilinear'
        self.output_mode = output_mode
        # ----- Encoder: Conv + LeakyReLU, stride=2 -----
        enc = []
        in_c = 1
        ch = self.base
        for _ in range(n_down):
            enc += [
                nn.Conv2d(in_c, ch, kernel_size=3, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = ch
            ch = min(ch * 2, 512)
        self.encoder = nn.Sequential(*enc)
        self.enc_out_ch = in_c  # channels after last block

        # FC layers initialized after we see the first input
        self.fc_mu = None
        self.fc_logvar = None
        self.fc_decode = None
        self._fc_shape = None  # (deepest_h, deepest_w)

        # ----- Decoder refine convs (applied after each upsample) -----
        self.dec_refine = nn.ModuleList()
        chs = [self.enc_out_ch]
        for _ in range(n_down - 1):
            chs.append(max(chs[-1] // 2, self.base))
        for i in range(len(chs) - 1):
            self.dec_refine.append(
                nn.Sequential(
                    nn.Conv2d(chs[i], chs[i+1], kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        # Output head: choose activation by output_mode
        if output_mode == "minus_one_one":
            act = nn.Tanh()
        elif output_mode == "zero_one":
            act = nn.Sigmoid()
        elif output_mode == "none":
            act = nn.Identity()
        else:
            raise ValueError("output_mode must be 'minus_one_one' | 'zero_one' | 'none'.")

        self.out_head = nn.Sequential(
            nn.Conv2d(self.base, 1, kernel_size=3, padding=1),
            act
        )

        self._enc_shapes = None  # [(H0,W0), (H1,W1), ..., (Hn,Wn)]
        self._init_weights()

    def _init_weights(self):
        # Xavier for convs, Kaiming for LeakyReLU can also be fine; either is OK here.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _lazy_fc_init(self, deepest_h, deepest_w, device):
        flat = self.enc_out_ch * deepest_h * deepest_w
        self.fc_mu = nn.Linear(flat, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.latent_dim).to(device)
        self.fc_decode = nn.Linear(self.latent_dim, flat).to(device)
        self._fc_shape = (deepest_h, deepest_w)

    def encode(self, x):
        # record shapes after each conv+act
        self._enc_shapes = []
        H, W = x.shape[-2:]
        self._enc_shapes.append((H, W))
        z = x
        for layer in self.encoder:
            z = layer(z)
            if isinstance(layer, nn.LeakyReLU):
                self._enc_shapes.append((z.shape[2], z.shape[3]))

        deepest_h, deepest_w = self._enc_shapes[-1]
        if self.fc_mu is None:
            self._lazy_fc_init(deepest_h, deepest_w, x.device)
        else:
            # safety: ensure consistent spatial size across batches
            if self._fc_shape != (deepest_h, deepest_w):
                raise RuntimeError(
                    f"VAE received different input size later in training. "
                    f"First FC was built for {self._fc_shape}, but got {(deepest_h, deepest_w)}."
                )

        z_flat = torch.flatten(z, start_dim=1)
        mu = self.fc_mu(z_flat)
        log_var = self.fc_logvar(z_flat)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        assert self._enc_shapes is not None and len(self._enc_shapes) >= 2, \
            "Call encode() before decode(), shapes missing."
        deepest_h, deepest_w = self._enc_shapes[-1]
        B = z.size(0)
        x = self.fc_decode(z).view(B, self.enc_out_ch, deepest_h, deepest_w)

        # Upsample step by step back to input size
        # enc_shapes: [(H0,W0), (H1,W1), ..., (Hn,Wn)]
        for i, target in enumerate(reversed(self._enc_shapes[:-1])):
            th, tw = target
            x = F.interpolate(x, size=(th, tw), mode=self.upsample_mode, align_corners=False if self.upsample_mode=='bilinear' else None)
            if i < len(self.dec_refine):
                x = self.dec_refine[i](x)

        return self.out_head(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


def vae_loss(recon_x, x, mu, log_var, kl_weight=1.0, reduction="mean"):
    mse = F.mse_loss(recon_x, x, reduction=reduction)
    if reduction == "sum":
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    else:
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return mse + kl_weight * kld, mse, kld

def train_vae(
    dataloader,
    model,
    device,
    optimizer,
    num_epochs=20,
    kl_weight=1.0,
    loss_reduction="mean",      # 'mean' or 'sum'
    use_amp=False,              # mixed precision
    grad_clip=1.0,
    save_path="vae_model_leaky_tanh.pth",
    log_every=30,
    save_best=True
):
    model.to(device)
    model.train()

    use_amp = bool(use_amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    history = []
    best_avg_loss = float("inf")

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse  = 0.0
        total_kld  = 0.0
        n_seen     = 0

        for batch_idx, batch in enumerate(dataloader):
            # dataset yields: ((x,y), tensor)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                _, data = batch
            else:
                data = batch[0] if isinstance(batch, (list, tuple)) else batch

            bs = data.size(0)
            n_seen += bs

            data = data.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    recon, mu, log_var = model(data)
                    loss, mse_loss, kl_loss = vae_loss(
                        recon, data, mu, log_var,
                        kl_weight=kl_weight,
                        reduction=loss_reduction
                    )
                # backward on scaled loss
                scaler.scale(loss).backward()
                # unscale before clipping
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, mu, log_var = model(data)
                loss, mse_loss, kl_loss = vae_loss(
                    recon, data, mu, log_var,
                    kl_weight=kl_weight,
                    reduction=loss_reduction
                )
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            # accumulate epoch totals
            if loss_reduction == "sum":
                total_loss += loss.item()
                total_mse  += mse_loss.item()
                total_kld  += kl_loss.item()
            else:  # 'mean' per-batch -> weight by batch size
                total_loss += loss.item() * bs
                total_mse  += mse_loss.item() * bs
                total_kld  += kl_loss.item() * bs

            if (batch_idx % log_every) == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch {batch_idx+1}  "
                    f"loss/b: {loss.item():.4f}  mse/b: {mse_loss.item():.4f}  kld/b: {kl_loss.item():.4f}")

        if n_seen == 0:
            print(f"Epoch {epoch+1}: no samples seen.")
            continue

        avg_loss = total_loss / n_seen
        avg_mse  = total_mse  / n_seen
        avg_kld  = total_kld  / n_seen

        history.append({"epoch": epoch+1, "avg_loss": avg_loss, "avg_mse": avg_mse, "avg_kld": avg_kld})
        print(f"====> Epoch: {epoch+1}  avg_loss: {avg_loss:.4f}  (mse: {avg_mse:.4f}, kld: {avg_kld:.4f})")

        if save_best and avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"[best] Saved model to {save_path} (avg_loss={avg_loss:.4f})")

    if not save_best:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return history





