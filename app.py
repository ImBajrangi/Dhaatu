import os
import tempfile
import gradio as gr
import torch
import numpy as np
import trimesh
from core.inference import TrellisInferencePipeline

# Zero-GPU support
try:
    import spaces
    _GPU_DECORATOR = spaces.GPU
except ImportError:
    def _GPU_DECORATOR(f): return f

# Initialize the pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = TrellisInferencePipeline(device=device)

@_GPU_DECORATOR
def generate_3d(image):
    if image is None:
        return None, "Status: No image provided"
    
    # Save input image to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    print(f"Generating Professional 3D Asset for {img_path}...")
    
    # Run the professional Hunyuan3D-2.1 pipeline
    result, _ = pipeline.generate(img_path)
    
    if isinstance(result, str) and os.path.exists(result):
        status = "✅ Successfully generated professional 3D asset using Hunyuan3D-2.1!"
        return result, status
    
    # Fallback to PoC cube if professional generation fails locally
    b1 = trimesh.creation.box(extents=[1, 1, 1])
    mesh = trimesh.util.concatenate([b1])
    out_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    mesh.export(out_path)
    
    error_msg = pipeline.real_model_error if hasattr(pipeline, 'real_model_error') else "Resource limit exceeded"
    status = f"⚠️ Mode: Local Lite (Reason: {error_msg})"
    return out_path, status

# Professional Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Dhaatu Professional: Image-to-3D Asset Factory")
    gr.Markdown("Transform high-resolution images into production-ready 3D models using **Hunyuan3D-2.1**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Image")
            run_btn = gr.Button("🚀 Generate Production 3D Model", variant="primary")
            gr.Examples(examples=["demo/logo_example.png", "demo/char_example.png"], inputs=input_img)
        
        with gr.Column(scale=2):
            status_out = gr.Markdown("Status: Initialize Pipeline...")
            output_3d = gr.Model3D(label="3D Production Viewer", clear_color=(0,0,0,0))
            
    run_btn.click(
        fn=generate_3d, 
        inputs=input_img, 
        outputs=[output_3d, status_out]
    )

if __name__ == "__main__":
    demo.launch()
