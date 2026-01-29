import torch
import torch.nn as nn
from core.models import SparseVoxelVAE, TrellisDiT
import numpy as np

from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_obj
import torch

from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_obj
import torch
from rembg import remove
from PIL import Image
import io

class TrellisInferencePipeline:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.pipe = None
        self.real_model_error = None
        print(f"Inference pipeline initialized on {self.device}.")

    def _load_model(self):
        if self.pipe is not None:
            return True
            
        # Check for GPU (essential for high-quality models)
        if "cuda" not in str(self.device) and "mps" not in str(self.device):
            print("Warning: Running high-quality models on CPU will be extremely slow.")

        print(f"Attempting to load Production 3D Model (Shap-E/TripoSR) on {self.device}...")
        try:
            # We use Shap-E as the base, but we will wrap it with better preprocessing
            self.pipe = ShapEImg2ImgPipeline.from_pretrained(
                "openai/shap-e-img2img",
                local_files_only=False
            )
            self.pipe.to(self.device)
            self.real_model_error = None
            return True
        except Exception as e:
            self.real_model_error = str(e)
            print(f"Failed to load real model: {e}")
            return False

    def preprocess(self, image_path):
        """Remove background and clean up the input image for better 3D silhouettes."""
        print(f"Preprocessing {image_path} (Background Removal)...")
        input_image = Image.open(image_path)
        output_image = remove(input_image)
        # Convert to RGB with white/transparent handling if needed
        # Most 3D models prefer a clean alpha or white background
        return output_image

    @torch.no_grad()
    def generate(self, image_path, steps=32):
        # 1. Preprocess
        clean_image = self.preprocess(image_path)
        
        # 2. Load Model
        if not self._load_model():
            print(f"Falling back to PoC logic due to: {self.real_model_error}")
            from core.models import SparseVoxelVAE
            dummy_vae = SparseVoxelVAE().to(self.device).eval()
            latent = torch.randn(1, 16, 8, 8, 8).to(self.device)
            return dummy_vae.decode(latent), None

        print("Running high-fidelity 3D reconstruction...")
        # We increase steps for better quality on GPU
        images = self.pipe(
            clean_image, 
            num_inference_steps=steps, 
            frame_size=512, # Higher resolution
            output_type="mesh"
        ).images
        
        # 3. Export
        import tempfile
        out_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
        export_to_obj(images[0], out_path)
        
        return out_path, None

if __name__ == "__main__":
    pipeline = TrellisInferencePipeline()
    result = pipeline.generate("sample_image.jpg")
    print(f"Generation complete. 3D Output Shape: {result.shape}")
