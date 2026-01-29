import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm(self.conv1(x)))
        x = self.norm(self.conv2(x))
        return F.relu(x + residual)

class SparseVoxelVAE(nn.Module):
    """
    Enhanced Sparse Voxel VAE with Residual Blocks.
    """
    def __init__(self, in_channels=4, latent_dim=16, base_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            ResidualBlock3d(base_channels),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ResidualBlock3d(base_channels * 2),
            nn.Conv3d(base_channels * 2, latent_dim * 2, kernel_size=3, stride=1, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channels * 2, kernel_size=3, stride=1, padding=1),
            ResidualBlock3d(base_channels * 2),
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            ResidualBlock3d(base_channels),
            nn.ConvTranspose3d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class TrellisDiT(nn.Module):
    """
    Advanced Diffusion Transformer (DiT) for 3D Generation.
    Uses Cross-Attention for conditioning and 3D positional embeddings.
    """
    def __init__(self, latent_dim=16, cond_dim=1024, depth=6, n_heads=8):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, 512)
        self.cond_proj = nn.Linear(cond_dim, 512)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, 512)) # Max 512 voxels for PoC
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512, 
                nhead=n_heads, 
                dim_feedforward=1024, 
                batch_first=True
            ),
            num_layers=depth
        )
        
        self.final_proj = nn.Linear(512, latent_dim)

    def forward(self, x_latent, condition):
        B, C, D, H, W = x_latent.shape
        x = x_latent.view(B, C, -1).permute(0, 2, 1) # [B, N, C]
        
        x = self.latent_proj(x)
        cond = self.cond_proj(condition).unsqueeze(1) # [B, 1, 512]
        
        # Add basic positional embedding and condition
        x = x + cond + self.pos_embed[:, :x.size(1), :]
        
        x = self.transformer(x)
        
        out = self.final_proj(x)
        out = out.permute(0, 2, 1).view(B, -1, D, H, W)
        return out

if __name__ == "__main__":
    # Test initialization
    vae = SparseVoxelVAE()
    dit = TrellisDiT()
    
    print("Models initialized successfully.")
    
    # Dummy forward pass
    dummy_3d = torch.randn(1, 4, 32, 32, 32)
    decoded, mu, logvar = vae(dummy_3d)
    print(f"VAE Output Shape: {decoded.shape}")
    
    dummy_latent = torch.randn(1, 16, 8, 8, 8)
    dummy_cond = torch.randn(1, 1024)
    dit_out = dit(dummy_latent, dummy_cond)
    print(f"DiT Output Shape: {dit_out.shape}")
