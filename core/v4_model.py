import torch
import torch.nn as nn

class DhaatuV4Model(nn.Module):
    def __init__(self, latent_dim=512): # Use 512 as per Config in training script
        super(DhaatuV4Model, self).__init__()

        # Encoder: 3D-CNN
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), # 8 -> 4
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, latent_dim)
        )

        # Decoder: Transpose 3D-CNN
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), # 4 -> 8
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), # 8 -> 16
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder_fc(latent)
        x = x.view(-1, 128, 4, 4, 4)
        reconstruction = self.decoder(x)
        return reconstruction, latent

def load_v4_model(checkpoint_path, device="cpu", latent_dim=512):
    model = DhaatuV4Model(latent_dim)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
