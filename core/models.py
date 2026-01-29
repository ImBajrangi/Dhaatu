import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseVoxelVAE(nn.Module):
    """
    Simplified Sparse Voxel VAE inspired by TRELLIS.
    Encodes 3D geometry into a compact latent space.
    """
    def __init__(self, in_channels=4, latent_dim=16, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels * 2, latent_dim * 2, kernel_size=3, stride=1, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
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
    Transformer-based Diffusion Model (DiT) for Structured Latent Diffusion.
    """
    def __init__(self, latent_dim=16, cond_dim=1024, depth=12):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, 512)
        self.cond_proj = nn.Linear(cond_dim, 512)
        
        # Transformer blocks (simplified)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
            for _ in range(depth)
        ])
        
        self.final_proj = nn.Linear(512, latent_dim)

    def forward(self, x_latent, condition):
        # Flatten voxels for transformer processing
        B, C, D, H, W = x_latent.shape
        x = x_latent.view(B, C, -1).permute(0, 2, 1) # [B, N, C]
        
        x = self.latent_proj(x)
        c = self.cond_proj(condition).unsqueeze(1) # [B, 1, 512]
        
        # Simple injection of condition
        x = x + c
        
        for block in self.blocks:
            x = block(x)
            
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
