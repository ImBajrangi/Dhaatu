"""
Dhaatu: Your Own Image-to-3D Generator
A custom 3D mesh generator using depth estimation.
Works on CPU - completely free!
"""
import gradio as gr
import numpy as np
import tempfile
import os
from PIL import Image

from core.depth_to_3d import pipeline


def generate_3d(image, depth_scale, resolution, simplify_factor, remove_background, bg_threshold, volumetric, thickness, smooth_iterations, depth_boost, aggressive_cut, method, isolate_main_object):
    """Generate 3D mesh from input image."""
    if image is None:
        return None, None, "❌ Please upload an image"
    
    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Generate mesh
        print(f"Starting generation with scale={depth_scale}, res={resolution}, simplify={simplify_factor}")
        mesh = pipeline.generate(
            image, 
            depth_scale=depth_scale,
            output_resolution=resolution,
            simplify_factor=simplify_factor,
            remove_background=remove_background,
            bg_threshold=bg_threshold,
            volumetric=volumetric,
            thickness=thickness,
            smooth_iterations=smooth_iterations,
            depth_boost=depth_boost,
            aggressive_cut=aggressive_cut,
            method=method,
            isolate_main_object=isolate_main_object
        )
        
        # Export to GLB
        print(f"Generating GLB for mesh with {len(mesh.faces)} faces...")
        
        # Build path in a way that avoids permission issues
        # and ensure the directory exists
        out_dir = "temp_models"
        os.makedirs(out_dir, exist_ok=True)
        glb_path = os.path.join(out_dir, f"model_{next(tempfile._get_candidate_names())}.glb")
        
        mesh.export(glb_path)
        print(f"Mesh exported to {glb_path}")
        
        status = f"✅ Generated 3D model with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
        if len(mesh.faces) > 50000:
            status += "\n⚠️ Warning: Large mesh may be slow to render in browser."
            
        return glb_path, glb_path, status
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"CRITICAL ERROR in generate_3d: {error_msg}")
        return None, None, f"❌ Error: {str(e)}"


# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗿 Dhaatu: Image-to-3D Generator")
    gr.Markdown("### Powered by Depth Anything V2 (Small) • State-of-the-Art Depth Estimation")
    gr.Markdown("Convert any image into a solid, smooth 3D model! Works on CPU.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📸 Input Image", type="pil")
            
            with gr.Accordion("⚙️ Generation Settings", open=True):
                method = gr.Dropdown(
                    choices=["Volumetric (Advanced)", "Surface Extrusion (Fast)"],
                    value="Volumetric (Advanced)",
                    label="🧠 Reconstruction Engine"
                )
                
                depth_scale = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.4, step=0.05,
                    label="Depth Scale (how 3D it looks)"
                )
                resolution = gr.Slider(
                    minimum=128, maximum=512, value=256, step=64,
                    label="Resolution (higher = more detail, slower)"
                )
                simplify_factor = gr.Slider(
                    minimum=0.01, maximum=1.0, value=0.1, step=0.05,
                    label="Mesh Quality (lower % = smaller/faster files)"
                )
                
                with gr.Group():
                    remove_background = gr.Checkbox(
                        value=True, label="✂️ Remove Background (Extract Object)"
                    )
                    bg_threshold = gr.Slider(
                        minimum=0.0, maximum=0.5, value=0.1, step=0.01,
                        label="Background Threshold (cut depth below this)"
                    )
                
                with gr.Group():
                    volumetric = gr.Checkbox(
                        value=True, label="📦 Enable Volume (Make it Filled/Solid)"
                    )
                    thickness = gr.Slider(
                        minimum=0.05, maximum=0.5, value=0.2, step=0.05,
                        label="Thickness (amount of volume)"
                    )
                    smooth_iterations = gr.Slider(
                        minimum=0, maximum=10, value=2, step=1,
                        label="Smoothness (Laplacian iterations)"
                    )
                
                with gr.Accordion("✨ Advanced Quality", open=False):
                    depth_boost = gr.Slider(
                        minimum=1.0, maximum=3.0, value=1.2, step=0.1,
                        label="🚀 Depth Boost (Contrast)"
                    )
                    aggressive_cut = gr.Checkbox(
                        value=True, label="🧼 Smart Mask Clean (Removes noise)"
                    )
                    isolate_main_object = gr.Checkbox(
                        value=True, label="🎯 Isolate Main Object (Delete extra parts)"
                    )
            
            generate_btn = gr.Button("🚀 Generate 3D Model", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            status_text = gr.Markdown("Status: Ready to generate!")
            model_viewer = gr.Model3D(label="🎮 3D Model Viewer")
            download_file = gr.File(label="📥 Download GLB")
    
    # Tips
    gr.Markdown("### 💡 Tips")
    gr.Markdown("""
    - **Best results**: Objects with clear foreground/background separation
    - **Depth Scale**: Lower = flatter, Higher = more 3D depth
    - **Resolution**: 256 is good balance, 512 for more detail
    - **Mesh Quality**: 0.1 (10%) is usually sufficient and loads fast in the viewer
    """)
    
    # Event handlers
    generate_btn.click(
        fn=generate_3d,
        inputs=[
            input_image, depth_scale, resolution, simplify_factor, 
            remove_background, bg_threshold, volumetric, thickness, 
            smooth_iterations, depth_boost, aggressive_cut, method, isolate_main_object
        ],
        outputs=[model_viewer, download_file, status_text]
    )

if __name__ == "__main__":
    print("Starting Dhaatu - Your Own Image-to-3D Generator...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
