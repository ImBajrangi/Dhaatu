import torch
import numpy as np
from PIL import Image
from skimage import measure
import trimesh
from .v4_model import load_v4_model
import os

class V4Pipeline:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        
    def load(self, checkpoint_path=None):
        target_path = checkpoint_path if checkpoint_path else self.checkpoint_path
        
        # Reload if model is not loaded or if path has changed
        if self.model is None or target_path != self.checkpoint_path:
            print(f"Loading Dhaatu V4 generative weights from {target_path}...")
            self.model = load_v4_model(target_path, self.device)
            self.checkpoint_path = target_path
            print("V4 Model ready.")
            
    def generate_from_depth(self, depth_map, checkpoint_path=None, voxel_size=32, threshold=0.5):
        """
        Takes a 2D depth map and refines it through the V4 Generative model.
        """
        self.load(checkpoint_path)
        
        # 1. Convert depth to 32x32x32 voxel grid (Simple projection)
        # depth_map is expected to be normalized 0-1
        grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
        
        # Resize depth map to voxel size
        d_resized = Image.fromarray(depth_map).resize((voxel_size, voxel_size), Image.BILINEAR)
        d_array = np.array(d_resized)
        
        for y in range(voxel_size):
            for x in range(voxel_size):
                d_val = d_array[y, x]
                if d_val > 0.05: # Simple threshold
                    # Map depth to Z-index
                    # In our training script, Z was often derived from depth
                    z_idx = int(d_val * (voxel_size - 1))
                    grid[y, x, z_idx] = 1.0
                    
        # 2. Run Inference
        voxel_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            reconstructed_voxels, _ = self.model(voxel_tensor)
        
        reconstructed_voxels = reconstructed_voxels.squeeze().cpu().numpy()
        
        # 3. Convert back to Mesh
        if reconstructed_voxels.max() < threshold:
             # Fallback if prediction is empty
             threshold = reconstructed_voxels.max() * 0.8
             
        try:
            verts, faces, normals, values = measure.marching_cubes(reconstructed_voxels, level=threshold)
            
            # Normalize and center
            verts = (verts / (voxel_size - 1)) * 2.0 - 1.0
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.process()
            return mesh
        except Exception as e:
            print(f"V4 Marching Cubes failed: {e}")
            return None
