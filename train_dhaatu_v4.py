import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils_data import DataLoader, Dataset
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
import requests
from pathlib import Path

# =============================================================================
# DHAATU V4: INDUSTRIAL 3D TRAINING SCRIPT (BIG DATA READY)
# Designed for Google Colab. Supports ModelNet40, ShapeNet, etc.
# =============================================================================

class Config:
    # Dataset
    VOXEL_SIZE = 32  # 32x32x32 resolution
    DATASET_PATH = "data/ModelNet40" # Auto-downloaded
    DATASET_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    SKIP_VOXEL_CONVERSION = False # Set to True if you already have .npy files
    
    # Model
    LATENT_DIM = 512 # Increased for complex ShapeNet objects
    
    # Training
    EPOCHS = 100
    LEARNING_RATE = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "checkpoints"

# -----------------------------------------------------------------------------
# 1. Big Data Manager & Voxelizer
# -----------------------------------------------------------------------------
class DatasetManager:
    @staticmethod
    def download_modelnet():
        """Automatically downloads and extracts ModelNet40 (Approx 450MB)."""
        target_dir = Path("data")
        target_dir.mkdir(exist_ok=True)
        zip_path = target_dir / "ModelNet40.zip"
        
        if (target_dir / "ModelNet40").exists():
            print("✅ ModelNet40 already exists.")
            return str(target_dir / "ModelNet40")

        print(f"📥 Downloading ModelNet40 from {Config.DATASET_URL}...")
        response = requests.get(Config.DATASET_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("📂 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        print("✨ Dataset ready!")
        return str(target_dir / "ModelNet40")

class Voxelizer:
    """Converts 3D Meshes (.off, .obj, .stl) to Voxel Grids."""
    @staticmethod
    def process_mesh(path, size=32):
        try:
            mesh = trimesh.load(path)
            # Center and scale
            mesh.apply_translation(-mesh.centroid)
            mesh.apply_scale(0.8 / mesh.extents.max())
            
            # Voxelize
            voxels = mesh.voxelized(pitch=1.0/size).matrix
            
            # Pad or crop to exact size
            result = np.zeros((size, size, size), dtype=bool)
            s_y, s_x, s_z = voxels.shape
            dy, dx, dz = min(size, s_y), min(size, s_x), min(size, s_z)
            result[:dy, :dx, :dz] = voxels[:dy, :dx, :dz]
            
            return result
        except Exception as e:
            print(f"⚠️ Failed to process {path}: {e}")
            return np.zeros((size, size, size), dtype=bool)

class VoxelDataset(Dataset):
    def __init__(self, root_dir, voxel_size=32, max_samples=None):
        self.root_dir = root_dir
        self.voxel_size = voxel_size
        self.file_list = []
        
        print(f"🔍 Scanning for 3D models in {root_dir}...")
        valid_exts = ('.off', '.obj', '.stl', '.npy')
        
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    self.file_list.append(os.path.join(root, file))
                    if max_samples and len(self.file_list) >= max_samples:
                        break
        
        print(f"✅ Found {len(self.file_list)} models.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        
        if path.endswith('.npy'):
            voxel = np.load(path)
        else:
            # On-the-fly voxelization for meshes
            voxel = Voxelizer.process_mesh(path, self.voxel_size)
        
        return torch.from_numpy(voxel).float().unsqueeze(0)

# -----------------------------------------------------------------------------
# 2. Model Architecture (3D-CNN Autoencoder)
# -----------------------------------------------------------------------------
class DhaatuV4Model(nn.Module):
    def __init__(self, latent_dim=256):
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

# -----------------------------------------------------------------------------
# 3. Training Logic
# -----------------------------------------------------------------------------
def train():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # Step 1: Big Data Download
    print("🚀 Dhaatu V4 Data Pipeline Initializing...")
    dataset_path = DatasetManager.download_modelnet()
    
    # Step 2: Initialize components
    print(f"📂 Loading Big Data from: {dataset_path}")
    dataset = VoxelDataset(dataset_path, Config.VOXEL_SIZE)
    if len(dataset) == 0:
        print("❌ Dataset empty! Check download path.")
        return
        
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    
    model = DhaatuV4Model(Config.LATENT_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCELoss() # Binary Cross Entropy for Voxel occupancy
    
    print(f"Starting training on {Config.DEVICE}...")
    
    history = []
    
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for batch in pbar:
            batch = batch.to(Config.DEVICE)
            
            # Forward pass
            outputs, _ = model(batch)
            loss = criterion(outputs, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        history.append(avg_loss)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(Config.SAVE_DIR, f"dhaatu_v4_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
    # Save final model
    torch.save(model.state_dict(), "dhaatu_v4_final.pth")
    print("Training complete! Model saved as 'dhaatu_v4_final.pth'")
    
    # Plot history
    plt.figure()
    plt.plot(history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.savefig("loss_history.png")
    plt.show()

# -----------------------------------------------------------------------------
# 4. Utilities for Export and Inference
# -----------------------------------------------------------------------------
def voxel_to_mesh(voxels, threshold=0.5):
    """Convert voxel grid to trimesh for export."""
    from skimage import measure
    verts, faces, normals, values = measure.marching_cubes(voxels[0, 0], level=threshold)
    return trimesh.Trimesh(vertices=verts, faces=faces)

if __name__ == "__main__":
    train()
