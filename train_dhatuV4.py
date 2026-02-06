import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import zipfile
import requests
from pathlib import Path
import glob # Import glob for easier file searching

_TRIMESH_AVAILABLE = False
try:
    import trimesh
    from skimage import measure
    _TRIMESH_AVAILABLE = True
    print("✅ trimesh and skimage.measure imported successfully.")
except ImportError:
    print("⚠️ trimesh or scikit-image not found. Mesh voxelization will be disabled. Please install trimesh (`pip install trimesh`) and scikit-image (`pip install scikit-image`) or provide .npy voxel data.")

import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# DHAATU V4: INDUSTRIAL 3D TRAINING SCRIPT (BIG DATA READY)
# Designed for Google Colab. Supports ModelNet40, ShapeNet, etc.
# =============================================================================

class Config:
    # Dataset
    VOXEL_SIZE = 32  # 32x32x32 resolution
    DATASET_PATH = "ModelNet40" # Local directory
    DATASET_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    BATCH_SIZE = 32
    NUM_WORKERS = 0 # Reduced from 4 to 0 to prevent DataLoader worker crashes
    SKIP_VOXEL_CONVERSION = False # Set to True if you already have .npy files
    PRE_VOXELIZE_TO_NPY = True # New: Enable or disable pre-voxelization to .npy
    PRE_VOXELIZED_DATA_PATH = "data/modelnet40_npy" # New: Directory for pre-voxelized .npy files

    # Model
    LATENT_DIM = 512 # Increased for complex ShapeNet objects

    # Training
    EPOCHS = 100
    LEARNING_RATE = 2e-4
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "checkpoints"

# -----------------------------------------------------------------------------
# 1. Big Data Manager & Voxelizer
# -----------------------------------------------------------------------------
class DatasetManager:
    @staticmethod
    def download_modelnet():
        """Automatically downloads and extracts ModelNet40 (Approx 450MB)."""
        # Check for local dataset first
        if Path(Config.DATASET_PATH).exists():
             print(f"✅ Local dataset found at {Config.DATASET_PATH}")
             return Config.DATASET_PATH

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

    @staticmethod
    def convert_meshes_to_voxels(source_dir, target_dir, voxel_size=32):
        if not _TRIMESH_AVAILABLE:
            print("Skipping pre-voxelization: trimesh not available.")
            return

        Path(target_dir).mkdir(parents=True, exist_ok=True)
        print(f"⚙️ Pre-voxelizing meshes from {source_dir} to {target_dir}...")

        mesh_files = []
        for ext in ('.off', '.obj', '.stl'):
            mesh_files.extend(glob.glob(os.path.join(source_dir, '**', f'*{ext}'), recursive=True))

        if not mesh_files:
            print("No mesh files found for pre-voxelization.")
            return

        pbar = tqdm(mesh_files, desc="Voxelizing meshes")
        for mesh_path in pbar:
            relative_path = os.path.relpath(mesh_path, source_dir)
            npy_path = os.path.join(target_dir, relative_path).replace(os.path.splitext(relative_path)[1], '.npy')
            Path(npy_path).parent.mkdir(parents=True, exist_ok=True)

            if os.path.exists(npy_path):
                continue # Skip if already processed

            voxel_data = Voxelizer.process_mesh(mesh_path, voxel_size)
            if voxel_data is not None:
                np.save(npy_path, voxel_data)
        print(f"✅ Pre-voxelization complete. {len(mesh_files)} files processed.")

class Voxelizer:
    """Converts 3D Meshes (.off, .obj, .stl) to Voxel Grids."""
    @staticmethod
    def process_mesh(path, size=32):
        if not _TRIMESH_AVAILABLE:
            # This check should ideally be handled before calling process_mesh
            # But as a failsafe, it's here too.
            print(f"Skipping voxelization for {path}: trimesh not available.")
            return np.zeros((size, size, size), dtype=bool)
        try:
            mesh = trimesh.load(path, process=False) # process=False to avoid unnecessary processing for voxelization
            
            # Only try to fill holes if mesh is not watertight, can be slow
            # if not mesh.is_watertight:
            #     mesh.fill_holes()

            # Center and scale
            mesh.apply_translation(-mesh.centroid)
            mesh.apply_scale(0.8 / mesh.extents.max()) # Scale to fit within unit cube approximately

            # Voxelize
            # pitch determines the resolution. A higher pitch means smaller voxels, thus higher resolution.
            # For a 32x32x32 grid, if the object is scaled to fit in a 1x1x1 cube, pitch=1.0/size is correct.
            voxels = mesh.voxelized(pitch=1.0/size).matrix

            # Pad or crop to exact size
            result = np.zeros((size, size, size), dtype=bool)
            # Ensure source dimensions don't exceed target dimensions
            s_y, s_x, s_z = voxels.shape
            dy, dx, dz = min(size, s_y), min(size, s_x), min(size, s_z)

            # Calculate start coordinates for centering
            start_y = (size - dy) // 2
            start_x = (size - dx) // 2
            start_z = (size - dz) // 2

            result[start_y:start_y+dy, start_x:start_x+dx, start_z:start_z+dz] = voxels[:dy, :dx, :dz]

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
        
        # Determine which files to scan based on pre-voxelization setting
        if Config.PRE_VOXELIZE_TO_NPY and os.path.exists(Config.PRE_VOXELIZED_DATA_PATH):
            scan_path = Config.PRE_VOXELIZED_DATA_PATH
            valid_exts = ('.npy',)
            print(f"Using pre-voxelized data from {scan_path}")
        else:
            scan_path = root_dir
            valid_exts = ('.npy',)
            if _TRIMESH_AVAILABLE:
                valid_exts += ('.off', '.obj', '.stl')
            

        for root, _, files in os.walk(scan_path):
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

        if path.lower().endswith('.npy'):
            voxel = np.load(path)
        elif _TRIMESH_AVAILABLE:
            # On-the-fly voxelization for meshes
            voxel = Voxelizer.process_mesh(path, self.voxel_size)
        else:
            # Fallback if trimesh is not available and it's not an npy file
            print(f"Skipping {path}: trimesh not available and not a .npy file. Returning empty voxel.")
            voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=bool)

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

    # Step 1.5: Pre-voxelization if enabled
    if Config.PRE_VOXELIZE_TO_NPY and not Config.SKIP_VOXEL_CONVERSION:
        # Check if target_dir has .npy files already, if so skip pre-voxelization
        # Fast check for existing voxel data
        has_voxels = False
        if os.path.exists(Config.PRE_VOXELIZED_DATA_PATH):
            # If directory exists and has subdirectories, assume it's pre-voxelized
            if any(os.path.isdir(os.path.join(Config.PRE_VOXELIZED_DATA_PATH, d)) for d in os.listdir(Config.PRE_VOXELIZED_DATA_PATH)):
                has_voxels = True

        if not has_voxels:
            DatasetManager.convert_meshes_to_voxels(dataset_path, Config.PRE_VOXELIZED_DATA_PATH, Config.VOXEL_SIZE)
        else:
            print(f"✅ Pre-voxelized data already found in {Config.PRE_VOXELIZED_DATA_PATH}, skipping pre-voxelization.")

    # Step 2: Initialize components
    print(f"📂 Loading Big Data from: {dataset_path}")
    dataset = VoxelDataset(dataset_path, Config.VOXEL_SIZE) # VoxelDataset will handle path selection
    if len(dataset) == 0:
        print("❌ Dataset empty! Check download path or install trimesh for mesh processing.")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE != "cpu")
    )

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
    if not _TRIMESH_AVAILABLE:
        print("trimesh not available, cannot convert voxel to mesh.")
        return None

    verts, faces, normals, values = measure.marching_cubes(voxels[0, 0], level=threshold)
    return trimesh.Trimesh(vertices=verts, faces=faces)

if __name__ == "__main__":
    train()
