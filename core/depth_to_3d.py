"""
Dhaatu: Custom Image-to-3D Converter
Uses depth estimation to generate 3D meshes from single images.
Works on CPU - no GPU required!
"""
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import trimesh
import os
from skimage import measure, morphology
import scipy.ndimage as ndimage

# Disable transformers background safetensors conversion check which can cause ReadTimeouts
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HUGGINGFACE_HUB_READ_TIMEOUT"] = "300" # Increase timeout just in case

class DepthTo3DPipeline:
    """
    Custom pipeline that converts images to 3D using depth estimation.
    Uses Depth Anything V2 (Small) which provides state-of-the-art results on CPU.
    """
    
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load the depth estimation model."""
        if self.model is not None:
            return
            
        print("Loading Depth Anything V2 Small model...")
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
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
        
        # Interpolate and normalize depth
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Normalize depth
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth.cpu().numpy()

    def clean_mask(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Perform morphological opening to remove small noise/filaments."""
        print(f"Cleaning mask with morphological operations ({iterations} iterations)...")
        # Ensure mask is boolean
        mask_bool = mask.astype(bool)
        
        # Remove small objects
        mask_clean = morphology.remove_small_objects(mask_bool, min_size=64)
        
        # Opening (erosion followed by dilation) to remove thin connections
        structure = np.ones((3, 3))
        mask_clean = ndimage.binary_opening(mask_clean, structure=structure, iterations=iterations)
        
        return mask_clean.astype(float)
    
    def smooth_depth_map(self, depth: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Apply a simple box blur to smooth the depth map."""
        if iterations <= 0:
            return depth
            
        # Convert to torch for fast convolution
        t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        
        for _ in range(iterations):
            # 3x3 mean filter
            kernel = torch.ones((1, 1, 3, 3), device=t.device) / 9.0
            t = torch.nn.functional.pad(t, (1, 1, 1, 1), mode='reflect')
            t = torch.nn.functional.conv2d(t, kernel)
            
        return t.squeeze().numpy()
    
    def depth_to_mesh(self, image: Image.Image, depth: np.ndarray, 
                       depth_scale: float = 0.5, simplify_factor: float = 0.1,
                       remove_background: bool = False, bg_threshold: float = 0.1,
                       volumetric: bool = True, thickness: float = 0.1,
                       smooth_iterations: int = 2, aggressive_cut: bool = True) -> trimesh.Trimesh:
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
            mask = depth.flatten()[faces] > bg_threshold
            
            if aggressive_cut:
                # Keep faces where ALL vertices are foreground (aggressive)
                keep_faces = np.all(mask, axis=1)
            else:
                # Keep faces where at least one vertex is foreground (lenient)
                keep_faces = np.any(mask, axis=1)
                
            faces = faces[keep_faces]
            print(f"Background removal: Reduced faces from {len(mask)} to {len(faces)}")
            
            # Re-index vertices to remove unused ones
            # This is important for find_boundary_edges
            unique_v_idx = np.unique(faces)
            v_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_v_idx)}
            vertices = vertices[unique_v_idx]
            faces = np.vectorize(v_mapping.get)(faces)
            
            # Also re-index vertex colors
            img_array = np.array(image.resize((width, height)))
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            vertex_colors = img_array.reshape(-1, 3)[unique_v_idx]
        else:
            # Get vertex colors from image (full grid)
            img_array = np.array(image.resize((width, height)))
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            vertex_colors = img_array.reshape(-1, 3)
        
        # Create initial mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors
        )
        
        # Add volume (extrusion)
        if volumetric:
            print(f"Adding volume (thickness={thickness})...")
            # Create a flat back face
            # We use the same vertices but with a constant Z
            back_vertices = vertices.copy()
            # Set Z to a fixed value behind the object
            back_vertices[:, 2] = depth_scale + thickness
            
            # Boundary edges connect front to back
            # We find edges that only belong to one face
            edges = mesh.edges_sorted
            unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
            boundary_edges = unique_edges[counts == 1]
            
            front_v_count = len(vertices)
            
            # Create side faces
            side_faces = []
            for edge in boundary_edges:
                v1, v2 = edge
                # Quad between (v1, v2) on front and (v1', v2') on back
                # v1' = v1 + front_v_count, v2' = v2 + front_v_count
                side_faces.append([v1, v2, v1 + front_v_count])
                side_faces.append([v2, v2 + front_v_count, v1 + front_v_count])
            
            # Create back faces (same as front but reversed)
            back_faces = faces[:, ::-1] + front_v_count
            
            # Combine all
            all_vertices = np.vstack([vertices, back_vertices])
            all_faces = np.vstack([faces, side_faces, back_faces])
            
            # Duplicate colors for back
            all_colors = np.vstack([vertex_colors, vertex_colors])
            
            mesh = trimesh.Trimesh(
                vertices=all_vertices,
                faces=all_faces,
                vertex_colors=all_colors
            )
        
        # Smoothing
        if smooth_iterations > 0:
            print(f"Applying Laplacian smoothing ({smooth_iterations} iterations)...")
            # Using volume_constraint=False to avoid "invalid value encountered in scalar power"
            # which happens when the mesh volume is zero or near-zero during smoothing.
            # We also use a smaller lamb for more stable smoothing.
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations, volume_constraint=False, lamb=0.5)
            except Exception as e:
                print(f"⚠️ Smoothing failed: {e}. Skipping smoothing step.")
        
        # Center the mesh
        mesh.vertices -= mesh.bounds.mean(axis=0)
        
        return mesh

    def simplify_quadric_decimation(self, mesh: trimesh.Trimesh, simplify_factor: float) -> trimesh.Trimesh:
        """Helper to simplify mesh with error handling."""
        try:
            # Calculate target face count
            current_faces = len(mesh.faces)
            target_faces = int(current_faces * simplify_factor)
            
            # Only simplify if we have a significant number of faces
            if current_faces > 1000 and target_faces < current_faces:
                print(f"Simplifying mesh from {current_faces} to {target_faces} faces...")
                try:
                    mesh = mesh.simplify_quadric_decimation(target_faces)
                except Exception as e:
                    if "target_reduction" in str(e):
                        reduction = 1.0 - simplify_factor
                        print(f"Retrying simplification with reduction={reduction:.2f}")
                        mesh = mesh.simplify_quadric_decimation(reduction)
                    else:
                        raise e
        except Exception as e:
            print(f"⚠️ Mesh simplification failed: {e}. Returning original mesh.")
        return mesh

    def depth_to_volumetric_mesh(self, image: Image.Image, depth: np.ndarray, 
                                 depth_scale: float = 0.5, thickness: float = 0.2,
                                 grid_res: int = 128, smooth_iterations: int = 2) -> trimesh.Trimesh:
        """
        Create a watertight manifold mesh using Marching Cubes.
        """
        print(f"Creating volumetric grid ({grid_res}x{grid_res}x{grid_res})...")
        # Padding of 1 to ensure a closed surface
        grid = np.zeros((grid_res + 2, grid_res + 2, grid_res + 2), dtype=float)
        
        # Resize depth to grid resolution
        depth_resized = Image.fromarray(depth).resize((grid_res, grid_res), Image.BILINEAR)
        depth_map = np.array(depth_resized)
        
        # Fill grid
        # Front face at z = depth_map * depth_scale
        # Back face at z = depth_scale + thickness
        # Map depth_scale + thickness to grid_res
        z_max_idx = grid_res
        z_scale = z_max_idx / (depth_scale + thickness + 1e-8)
        
        # Boundary Trimming: We ignore the very edges of the grid (1-pixel border inside padding)
        # to prevent background "walls" from forming at the image boundaries.
        trim = 1
        
        for y in range(trim, grid_res - trim):
            for x in range(trim, grid_res - trim):
                d = depth_map[y, x]
                if d <= 0: continue # Skip masked background
                
                z_front = d * depth_scale
                z_back = depth_scale + thickness
                
                z_start = int(z_front * z_scale) + 1
                z_end = int(z_back * z_scale) + 1
                
                # Further boundary check: if d is close to 0 but we passed the mask, 
                # maybe force it to zero if it's very close to the trim border?
                # For now, traditional fill.
                grid[y+1, x+1, z_start:z_end+1] = 1.0
        
        print("Running Marching Cubes...")
        # marching_cubes returns (axis0, axis1, axis2)
        try:
            verts, faces, normals, values = measure.marching_cubes(grid, level=0.5)
        except Exception as e:
            print(f"❌ Marching Cubes failed: {e}. Falling back to simple extrusion.")
            # This can happen if the grid is empty
            return self.depth_to_mesh(image, depth, depth_scale, simplify_factor=1.0)
            
        v_final = np.zeros_like(verts)
        v_final[:, 0] = (verts[:, 1] - 1) / grid_res # X
        v_final[:, 1] = (verts[:, 0] - 1) / grid_res # Y
        v_final[:, 2] = (verts[:, 2] - 1) / z_scale  # Z world space
        
        v_final[:, 1] = 1.0 - v_final[:, 1] # Flip Y
        
        print("Mapping vertex colors...")
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        px = np.clip(v_final[:, 0] * (w - 1), 0, w - 1).astype(int)
        py = np.clip(v_final[:, 1] * (h - 1), 0, h - 1).astype(int)
        vertex_colors = img_array[py, px]
        
        mesh = trimesh.Trimesh(vertices=v_final, faces=faces, vertex_colors=vertex_colors)
        
        if smooth_iterations > 0:
            print(f"Refining surface ({smooth_iterations} iterations)...")
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations, volume_constraint=False)
            except: pass
            
        mesh.vertices -= mesh.bounds.mean(axis=0) # Center
        return mesh

    def isolate_largest_component(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Keep only the largest connected component of the mesh."""
        print("Isolating main object (removing floating artifacts)...")
        try:
            components = mesh.split(only_watertight=False)
            if not components:
                return mesh
            
            # Find the component with the largest volume or surface area
            # Volume is better but can be zero for non-watertight meshes
            largest_component = max(components, key=lambda m: m.area)
            print(f"Isolated main object: {len(largest_component.vertices)} vertices (removed {len(components)-1} fragments)")
            return largest_component
        except Exception as e:
            print(f"⚠️ Isolation failed: {e}. Returning full mesh.")
            return mesh
    
    def generate(self, image: Image.Image, depth_scale: float = 0.5, 
                 output_resolution: int = 256, simplify_factor: float = 0.1,
                 remove_background: bool = True, bg_threshold: float = 0.1,
                 volumetric: bool = True, thickness: float = 0.2,
                 smooth_iterations: int = 2, depth_boost: float = 1.2,
                 aggressive_cut: bool = True, method: str = "Volumetric (Advanced)",
                 isolate_main_object: bool = True) -> trimesh.Trimesh:
        """
        Full pipeline: Image -> Depth -> 3D Mesh
        
        Args:
            image: Input PIL Image
            depth_scale: Depth extrusion amount (0.1-1.0)
            output_resolution: Resolution for processing (lower = faster)
            simplify_factor: How much to simplify (0.1-1.0)
            remove_background: Whether to remove low-depth areas
            bg_threshold: Depth threshold for removal (0-1)
            volumetric: Whether to add a back face and sides (for Surface method)
            thickness: How thick the model should be
            smooth_iterations: Number of smoothing passes
            depth_boost: Multiply depth values to enhance detail (1.0-2.0)
            aggressive_cut: Whether to be more aggressive with background removal
            method: "Surface Extrusion" or "Volumetric (Advanced)"
            isolate_main_object: Whether to keep only the largest connected part
        """
        # Resize for processing
        image_resized = image.resize((output_resolution, output_resolution))
        
        # Convert to RGB if needed
        if image_resized.mode != 'RGB':
            image_resized = image_resized.convert('RGB')
        
        print(f"Estimating depth at {output_resolution}x{output_resolution}...")
        depth = self.estimate_depth(image_resized)
        
        # Apply depth boost (contrast)
        if depth_boost > 1.0:
            print(f"Boosting depth contrast (x{depth_boost})...")
            depth = np.clip(depth * depth_boost, 0, 1)
        
        # Smooth depth map to reduce staircase artifacts
        print("Smoothing depth map...")
        depth = self.smooth_depth_map(depth, iterations=1)
        
        # Mask the depth map
        if remove_background:
            mask = depth > bg_threshold
            
            # Clean the mask if aggressive_cut is on (acting as "Smart Cleanup")
            if aggressive_cut:
                mask = self.clean_mask(mask, iterations=1)
                
            depth[~mask.astype(bool)] = 0
            
        if "Volumetric" in method:
            print(f"Generating high-fidelity 3D model using Marching Cubes...")
            mesh = self.depth_to_volumetric_mesh(
                image_resized,
                depth,
                depth_scale=depth_scale,
                thickness=thickness,
                grid_res=output_resolution // 2, # Balance quality and speed
                smooth_iterations=smooth_iterations
            )
        else:
            print(f"Generating 3D model using Surface Extrusion...")
            mesh = self.depth_to_mesh(
                image_resized, 
                depth, 
                depth_scale=depth_scale, 
                simplify_factor=simplify_factor,
                remove_background=remove_background,
                bg_threshold=bg_threshold,
                volumetric=volumetric,
                thickness=thickness,
                smooth_iterations=smooth_iterations,
                aggressive_cut=aggressive_cut
            )
        
        # Isolate main object if requested
        if isolate_main_object:
            mesh = self.isolate_largest_component(mesh)
            
        # Final simplification (if not already simplified in depth_to_mesh)
        if 0.01 <= simplify_factor < 1.0:
            mesh = self.simplify_quadric_decimation(mesh, simplify_factor)
            
        print("3D mesh generated successfully!")
        return mesh


# Create singleton instance
pipeline = DepthTo3DPipeline()
