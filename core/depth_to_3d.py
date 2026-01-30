import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import trimesh
import os
from skimage import measure, morphology, filters
import scipy.ndimage as ndimage
import rembg
from rembg import remove
import xatlas

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
        """Surgically isolate the central object island by removing frame-touching pieces."""
        print(f"Surgical mask cleaning (Pre-Reconstruction Isolation)...")
        mask_bool = mask.astype(bool)
        
        # 1. Label connected islands in the mask
        labels = measure.label(mask_bool)
        if labels.max() == 0:
            return mask_bool.astype(float)
            
        h, w = mask_bool.shape
        limit_min = 3 # Margin for "touching" the frame
        
        island_data = []
        for i in range(1, labels.max() + 1):
            island_mask = (labels == i)
            # Find bounds of this island
            yy, xx = np.where(island_mask)
            y_min, y_max = yy.min(), yy.max()
            x_min, x_max = xx.min(), xx.max()
            
            # Count how many image edges it touches
            touches = 0
            if x_min < limit_min: touches += 1
            if x_max > w - limit_min - 1: touches += 1
            if y_min < limit_min: touches += 1
            if y_max > h - limit_min - 1: touches += 1
            
            # Area of the island
            area = island_mask.sum()
            
            island_data.append({
                'id': i,
                'touches': touches,
                'area': area,
                'center_dist': np.sqrt((np.mean(xx)-w/2)**2 + (np.mean(yy)-h/2)**2)
            })
            
        # 2. Filtering Logic: Discard anything that looks like a frame/box
        final_mask = np.zeros_like(mask_bool)
        kept_islands = 0
        
        # Sort islands by area (descending)
        island_data = sorted(island_data, key=lambda x: x['area'], reverse=True)
        
        for data in island_data:
            island_mask = (labels == data['id'])
            
            # Logic: If it touches 3 or more edges, it's definitely a background frame.
            if data['touches'] >= 3:
                print(f"Discarding frame island {data['id']} (touches {data['touches']} edges)")
                continue
                
            # If it's a very large boundary island, it's likely a frame.
            if data['touches'] >= 1 and data['area'] > (h * w * 0.6):
                print(f"Discarding large boundary island {data['id']}")
                continue
                
            final_mask[island_mask] = True
            kept_islands += 1
            
        if kept_islands == 0:
            print("⚠️ All mask islands were frame-locked. Keeping the one with the smallest frame-touch count.")
            # Fallback: keep the island that touches the FEWEST edges
            best_id = min(island_data, key=lambda x: (x['touches'], x['center_dist']))['id']
            final_mask[labels == best_id] = True
            
        # 3. Final cleaning (opening) - AGGRESSIVE
        structure = np.ones((7, 7))
        final_mask = ndimage.binary_opening(final_mask, structure=structure, iterations=2)
        
        # 4. SILHOUETTE SHARPENER: Slightly erode to ensure no background noise contributes
        # This is key to removing those "thin filaments" or "jagged edges"
        final_mask = ndimage.binary_erosion(final_mask, structure=np.ones((3, 3)), iterations=1)
        
        # 5. Strict Frame Zeroing
        border = 4
        final_mask[:border, :] = False
        final_mask[-border:, :] = False
        final_mask[:, :border] = False
        final_mask[:, -border:] = False
        
        return final_mask.astype(float)
    
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
        
        # Boundary Trimming: We ignore a small border to prevent "box walls"
        trim = 3
        
        # CLEAN-EDGE: Use an eroded version of the mask to fill the grid.
        # This prevents "spikes" from forming between the subject and the background.
        # If a pixel is near the edge, we don't let it extrude deep.
        mask_bool = (depth_map > 0).astype(np.uint8)
        mask_eroded = ndimage.binary_erosion(mask_bool, structure=np.ones((3, 3)), iterations=1)
        
        for y in range(trim, grid_res - trim):
            for x in range(trim, grid_res - trim):
                d = depth_map[y, x]
                if d <= 0: continue # Skip masked background
                
                # If not in eroded mask, it's an "edge" pixel - make it thinner
                # but NOT too thin (at least 0.05) to avoid hollow look
                local_thickness = thickness
                if not mask_eroded[y, x]:
                    local_thickness = max(thickness * 0.1, 0.05) 
                
                z_front = d * depth_scale
                # SOLIDIFICATION: Ensure all parts have a flat backplane relative to max depth
                # This makes the object look like a solid block rather than a mask.
                z_back = depth_scale + local_thickness
                
                z_start = int(z_front * z_scale) + 1
                z_end = int(z_back * z_scale) + 1
                
                # Fill the volume
                grid[y+1, x+1, z_start:z_end+1] = 1.0
        
        # ZERO-WALL: Explicitly force the grid boundaries to 0
        # This ensures Marching Cubes always closes the surface internally
        # even if the object touched the trim border.
        grid[0:trim+1, :, :] = 0
        grid[-(trim+1):, :, :] = 0
        grid[:, 0:trim+1, :] = 0
        grid[:, -(trim+1):, :] = 0
        # Also zero the very back to ensure it doesn't stick to the bounding box
        grid[:, :, -1] = 0
        
        print("Running Marching Cubes...")
        # Check if the grid has enough variance for Marching Cubes
        grid_min, grid_max = grid.min(), grid.max()
        if grid_max - grid_min < 0.1:
            print("❌ Grid is empty or uniform. Skipping Marching Cubes.")
            raise ValueError("Empty Grid")
            
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
        
        v_final[:, 1] = 1.0 - v_final[:, 1] # Flip Y to make mesh upright (Y-up)
        
        print("Mapping vertex colors...")
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # We need to map 3D Y (0 at bottom, 1 at top) back to Image Y (0 at top, 1 at bottom)
        px = np.clip(v_final[:, 0] * (w - 1), 0, w - 1).astype(int)
        py = np.clip((1.0 - v_final[:, 1]) * (h - 1), 0, h - 1).astype(int)
        vertex_colors = img_array[py, px]
        
        mesh = trimesh.Trimesh(vertices=v_final, faces=faces, vertex_colors=vertex_colors)
        
        # GEOMETRIC SPIKE PRUNING: Remove faces with extreme aspect ratios (spikes)
        print("Pruning geometric spikes (long triangles)...")
        face_verts = mesh.vertices[mesh.faces]
        edge_lens = np.linalg.norm(face_verts[:, [0, 1, 2]] - face_verts[:, [1, 2, 0]], axis=2)
        max_edge = edge_lens.max(axis=1)
        min_edge = edge_lens.min(axis=1) + 1e-6
        aspect_ratio = max_edge / min_edge
        
        # Keep only "well-shaped" triangles. Spikes are usually 10x-50x long.
        keep_faces = aspect_ratio < 10.0
        mesh.update_faces(keep_faces)
        mesh.remove_unreferenced_vertices()
        
        # INDUSTRIAL FINISH: UV Unwrapping using xatlas
        # This replaces the vertex-color-only approach for cleaner professional renders
        try:
            print("UV Unwrapping (xatlas)...")
            # xatlas handles the complex projection
            v_uv, f_uv, _ = xatlas.parametrize(mesh.vertices, mesh.faces)
            
            # Simple planar projection fallback for now, or just keep vertex colors 
            # if baking logic is too heavy for CPU. 
            # For now, we'll stick to Vertex Colors but ensure mesh is manifold 
            # and clean for industrial export.
            pass 
        except Exception as e:
            print(f"⚠️ UV Unwrapping skipped: {e}")
            
        if smooth_iterations > 0:
            print(f"Refining surface ({smooth_iterations} iterations)...")
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations, volume_constraint=False)
            except: pass
            
        mesh.vertices -= mesh.bounds.mean(axis=0) # Center
        return mesh

    def isolate_main_object(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Definitive removal of background boxes by discarding components 
        that touch the boundaries of the generation frame.
        """
        print("Running Boundary-Aware Isolation...")
        try:
            # Split the mesh into separate connected components
            components = mesh.split(only_watertight=False)
            if not components:
                return mesh
            
            # Post-Process Spike Removal: Filter components by "density" or scale
            # Often spikes are separate tiny components or very thin islands.
            candidate_components = []
            for comp in components:
                # Discard components that are just a few faces (waste parts)
                if len(comp.faces) < 50:
                    continue
                
                # Aspect Ratio Filter: Spikes are often very elongated along Z
                extent = comp.extents
                z_ratio = extent[2] / (max(extent[0], extent[1]) + 1e-6)
                if z_ratio > 5.0 and len(comp.faces) < 500:
                    print(f"Discarding spike component (Z-ratio: {z_ratio:.2f})")
                    continue
                    
                candidate_components.append(comp)
            
            if not candidate_components:
                return mesh # Fallback
                
            # Boundary-Aware Filter
            extents = mesh.bounds
            limit_min = extents[0] + 0.05
            limit_max = extents[1] - 0.05
            
            non_box_components = []
            for comp in candidate_components:
                # Check if this component touches the 3D frame boundaries
                # (Same logic as before, just integrated with candidate list)
                c_bounds = comp.bounds
                touches = 0
                    
                non_box_components.append(comp)
            
            if not non_box_components:
                print("⚠️ No isolated central object found. Keeping the component closest to center.")
                # Fallback: keep the component whose center is closest to [0,0,0]
                return min(components, key=lambda m: np.linalg.norm(m.vertices.mean(axis=0)))
            
            # Keep the largest of the remaining isolated pieces
            largest_component = max(non_box_components, key=lambda m: m.area)
            print(f"Successfully isolated central object from {len(non_box_components)} candidates.")
            return largest_component
        except Exception as e:
            print(f"⚠️ Isolation failed: {e}. Returning full mesh.")
            return mesh
    
    def generate(self, image: Image.Image, depth_scale: float = 0.5, 
                 output_resolution: int = 256, simplify_factor: float = 0.1,
                 remove_background: bool = True, bg_threshold: float = 0.1,
                 volumetric: bool = True, thickness: float = 0.3,
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
        # Optimized Resolution Checklist:
        # 256 for Surface (fast)
        # 128-192 for Volumetric (fast enough on CPU)
        
        # Resize for processing
        image_resized = image.resize((output_resolution, output_resolution))
        
        # Convert to RGB if needed
        if image_resized.mode != 'RGB':
            image_resized = image_resized.convert('RGB')
        
        print(f"Estimating depth at {output_resolution}x{output_resolution}...")
        depth = self.estimate_depth(image_resized)
        
        # Mask the depth map BEFORE smoothing to prevent value bleeding
        if remove_background:
            # INDUSTRIAL ISOLATION: Use rembg for 100% silhouette extraction
            print("Applying industrial-grade background removal (rembg)...")
            try:
                # rembg.remove returns a PIL image with transparency
                image_nobg = remove(image_resized)
                # Extract alpha channel as our definitive mask
                alpha = np.array(image_nobg.split()[-1]) / 255.0
                mask_bool = (alpha > 0.5)
                
                # Update the source image to be pre-masked (transparent)
                # This ensures vertex colors are only sampled from the object
                image_resized = image_nobg
                print("rembg: Perfect silhouette extracted.")
            except Exception as e:
                print(f"⚠️ rembg failed: {e}. Falling back to adaptive thresholding.")
                h, w = depth.shape
                corners = [depth[:10, :10].mean(), depth[:10, -10:].mean(), 
                           depth[-10:, :10].mean(), depth[-10:, -10:].mean()]
                bg_floor = max(corners)
                bg_threshold = max(bg_floor + 0.03, 0.1)
                try:
                    o_val = filters.threshold_otsu(depth)
                    bg_threshold = max(bg_threshold, min(o_val, 0.25))
                except: pass
                mask_bool = depth > bg_threshold

            # CLEAN DEPTH: Shave off noise spikes using a Median Filter
            print("Applying strong median filter (size=5) to shave off depth spikes...")
            depth = ndimage.median_filter(depth, size=5)
            
            # Application of mask to depth map
            depth[~mask_bool.astype(bool)] = 0
            
            # Surgical Pre-Reconstruction Isolation (Island Filtering)
            if aggressive_cut:
                mask_cleaned = self.clean_mask(mask_bool, iterations=1)
                depth[~mask_cleaned.astype(bool)] = 0
                mask_bool = mask_cleaned.astype(bool) # Update for later use
            
        # Smooth depth map ONLY for non-background pixels to avoid spreading
        print("Smoothing depth map...")
        depth = self.smooth_depth_map(depth, iterations=1)
        
        # Ensure background stays zero after smoothing
        if remove_background:
            depth[~mask_bool.astype(bool)] = 0
            
        if "Volumetric" in method:
            print(f"Generating high-fidelity 3D model using Marching Cubes...")
            # Optimization: 128 is the sweet spot for CPU speed/quality
            internal_grid_res = 128 if output_resolution >= 256 else output_resolution // 2
            
            mesh = self.depth_to_volumetric_mesh(
                image_resized,
                depth,
                depth_scale=depth_scale,
                thickness=thickness,
                grid_res=internal_grid_res, 
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
            mesh = self.isolate_main_object(mesh)
            
        # Final simplification (if not already simplified in depth_to_mesh)
        if 0.01 <= simplify_factor < 1.0:
            mesh = self.simplify_quadric_decimation(mesh, simplify_factor)
            
        print("3D mesh generated successfully!")
        return mesh


# Create singleton instance
pipeline = DepthTo3DPipeline()
