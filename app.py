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


def generate_3d(image, depth_scale, resolution):
    """Generate 3D mesh from input image."""
    if image is None:
        return None, None, "❌ Please upload an image"
    
    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Generate mesh
        mesh = pipeline.generate(
            image, 
            depth_scale=depth_scale,
            output_resolution=resolution
        )
        
        # Export to GLB
        glb_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
        mesh.export(glb_path)
        
        # Also export OBJ for compatibility
        obj_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
        mesh.export(obj_path)
        
        status = f"✅ Generated 3D model with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
        return glb_path, glb_path, status
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🗿 Dhaatu: Your Own Image-to-3D Generator
    
    **100% Free • Runs on CPU • You Own It Completely**
    
    This converter uses depth estimation (Intel DPT) to create 3D meshes from single images.
    Upload any image and get a downloadable 3D model!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📸 Input Image", type="pil")
            
            with gr.Accordion("⚙️ Settings", open=True):
                depth_scale = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.4, step=0.05,
                    label="Depth Scale (how 3D it looks)"
                )
                resolution = gr.Slider(
                    minimum=128, maximum=512, value=256, step=64,
                    label="Resolution (higher = more detail, slower)"
                )
            
            generate_btn = gr.Button("🚀 Generate 3D Model", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            status_text = gr.Markdown("Status: Ready to generate!")
            model_viewer = gr.Model3D(label="🎮 3D Model Viewer", clear_color=(0.1, 0.1, 0.1, 1))
            download_btn = gr.DownloadButton(label="📥 Download GLB", interactive=False)
    
    # Examples
    gr.Markdown("### 💡 Tips")
    gr.Markdown("""
    - **Best results**: Objects with clear foreground/background separation
    - **Depth Scale**: Lower = flatter, Higher = more 3D depth
    - **Resolution**: 256 is good balance, 512 for more detail
    """)
    
    # Event handlers
    generate_btn.click(
        fn=generate_3d,
        inputs=[input_image, depth_scale, resolution],
        outputs=[model_viewer, download_btn, status_text]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn]
    )


# Launch
if __name__ == "__main__":
    print("Starting Dhaatu - Your Own Image-to-3D Generator...")
    demo.launch()
