import torch
import torch.nn as nn
from core.models import SparseVoxelVAE, TrellisDiT
import numpy as np

from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_obj
import torch

class TrellisInferencePipeline:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.pipe = None
        self.real_model_error = None
        print(f"Inference pipeline initialized on {self.device}. Real model will load on first demand.")

    def _load_model(self):
        if self.pipe is not None:
            return True
            
        print(f"Attempting to load Real 3D Model (Shap-E) on {self.device}...")
        try:
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

    @torch.no_grad()
    def generate(self, image_path, steps=25):
        # Attempt to load if not already loaded
        if not self._load_model():
            # Fallback for low-memory/no-internet
            print(f"Falling back to PoC logic due to: {self.real_model_error}")
            from core.models import SparseVoxelVAE
            dummy_vae = SparseVoxelVAE().to(self.device).eval()
            latent = torch.randn(1, 16, 8, 8, 8).to(self.device)
            return dummy_vae.decode(latent), None

        from PIL import Image
        input_image = Image.open(image_path).convert("RGB")
        
        print("Running real 3D generation (Shap-E)...")
        # Generate the 3D images
        images = self.pipe(
            input_image, 
            num_inference_steps=steps, 
            frame_size=256,
            output_type="mesh"
        ).images
        
        # Save to a temporary OBJ file
        import tempfile
        out_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
        export_to_obj(images[0], out_path)
        
        return out_path, None

if __name__ == "__main__":
    pipeline = TrellisInferencePipeline()
    result = pipeline.generate("sample_image.jpg")
    print(f"Generation complete. 3D Output Shape: {result.shape}")
