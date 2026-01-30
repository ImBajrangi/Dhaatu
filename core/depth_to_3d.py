"""
Dhaatu: Custom Image-to-3D Converter
Uses depth estimation to generate 3D meshes from single images.
Works on CPU - no GPU required!
"""
import numpy as np
from PIL import Image
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
import trimesh

class DepthTo3DPipeline:
    """
    Custom pipeline that converts images to 3D using depth estimation.
    Uses Intel's DPT model which runs efficiently on CPU.
    """
    
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load the depth estimation model."""
        if self.model is not None:
            return
            
        print("Loading DPT depth estimation model...")
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Estimate depth from a single image."""
        self.load_model()
        
        # Prepare image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict depth
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy
        depth = prediction.squeeze().cpu().numpy()
        
        # Normalize depth
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth
    
    def depth_to_mesh(self, image: Image.Image, depth: np.ndarray, 
                       depth_scale: float = 0.5, simplify_factor: float = 0.1,
                       remove_background: bool = False, bg_threshold: float = 0.1) -> trimesh.Trimesh:
        """
        Convert depth map to 3D mesh.
        
        Args:
            image: Original RGB image for texture
            depth: Depth map (normalized 0-1)
            depth_scale: How much to extrude the depth (higher = more 3D)
            simplify_factor: Keep percentage (0.1 = keep 10% of faces)
        """
        height, width = depth.shape
        
        # Create grid of vertices
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        # Z from depth (invert so closer objects are in front)
        zz = (1 - depth) * depth_scale
        
        # Create vertices
        vertices = np.stack([xx, -yy, zz], axis=-1).reshape(-1, 3)
        
        # Create faces (triangulate the grid)
        # Using a faster method for Grid creation
        i = np.arange(height - 1)
        j = np.arange(width - 1)
        ii, jj = np.meshgrid(i, j, indexing='ij')
        
        idx = ii * width + jj
        
        # Two triangles per quad
        f1 = np.stack([idx, idx + width, idx + 1], axis=-1).reshape(-1, 3)
        f2 = np.stack([idx + 1, idx + width, idx + width + 1], axis=-1).reshape(-1, 3)
        faces = np.vstack([f1, f2])
        
        # Background removal (optional)
        if remove_background:
            # depth ranges from 0 (far) to 1 (near)
            # We want to remove vertices where depth is near 0
            # A face is background if ALL its vertices are below the threshold
            mask = depth.flatten()[faces] > bg_threshold
            # Keep faces where at least one vertex is foreground
            keep_faces = np.any(mask, axis=1)
            faces = faces[keep_faces]
            
            print(f"Background removal: Reduced faces from {len(keep_faces)} to {len(faces)}")
        
        # Get vertex colors from image
        img_array = np.array(image.resize((width, height)))
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        
        vertex_colors = img_array.reshape(-1, 3)
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors
        )
        
        # Simplify mesh to reduce file size
        if 0.01 <= simplify_factor < 1.0:
            try:
                # Calculate target face count
                current_faces = len(mesh.faces)
                target_faces = int(current_faces * simplify_factor)
                
                # Only simplify if we have a significant number of faces
                # and target is strictly less than current
                if current_faces > 1000 and target_faces < current_faces:
                    print(f"Simplifying mesh from {current_faces} to {target_faces} faces...")
                    # Try both integer face_count and percentage as some trimesh versions/backends 
                    # have different expectations for the same method
                    try:
                        mesh = mesh.simplify_quadric_decimation(target_faces)
                    except Exception as e:
                        if "target_reduction" in str(e):
                            # Reduction factor for fast_simplification backend
                            reduction = 1.0 - simplify_factor
                            print(f"Retrying simplification with reduction={reduction:.2f}")
                            # We can try passing reduction if trimesh supports it via kwargs 
                            # or just try to pass it as the first argument
                            mesh = mesh.simplify_quadric_decimation(reduction)
                        else:
                            raise e
            except Exception as e:
                print(f"⚠️ Mesh simplification failed: {e}. Returning original mesh.")
                # We return the mesh anyway, it just might be a bit larger
        
        return mesh
    
    def generate(self, image: Image.Image, depth_scale: float = 0.5, 
                 output_resolution: int = 256, simplify_factor: float = 0.1,
                 remove_background: bool = False, bg_threshold: float = 0.1) -> trimesh.Trimesh:
        """
        Full pipeline: Image -> Depth -> 3D Mesh
        
        Args:
            image: Input PIL Image
            depth_scale: Depth extrusion amount (0.1-1.0)
            output_resolution: Resolution for processing (lower = faster)
            simplify_factor: How much to simplify (0.1-1.0)
            remove_background: Whether to remove low-depth areas
            bg_threshold: Depth threshold for removal (0-1)
        """
        # Resize for processing
        image_resized = image.resize((output_resolution, output_resolution))
        
        # Convert to RGB if needed
        if image_resized.mode != 'RGB':
            image_resized = image_resized.convert('RGB')
        
        print(f"Estimating depth at {output_resolution}x{output_resolution}...")
        depth = self.estimate_depth(image_resized)
        
        print(f"Generating 3D mesh (simplify={simplify_factor}, rm_bg={remove_background})...")
        mesh = self.depth_to_mesh(
            image_resized, 
            depth, 
            depth_scale=depth_scale, 
            simplify_factor=simplify_factor,
            remove_background=remove_background,
            bg_threshold=bg_threshold
        )
        
        print("3D mesh generated successfully!")
        return mesh


# Create singleton instance
pipeline = DepthTo3DPipeline()
