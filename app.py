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

    # Try Tripo3D Cloud API first (best quality)
    from core.tripo_api import tripo_client
    
    if tripo_client.is_configured():
        print("Using Tripo3D Cloud API for professional 3D generation...")
        result, error = tripo_client.generate(img_path)
        
        if result and os.path.exists(result):
            return result, "✅ Production-grade 3D model generated via Tripo3D Cloud!"
        else:
            print(f"Tripo3D API failed: {error}")
    else:
        print("TRIPO_API_KEY not set. Using local fallback...")
    
    # Fallback to local model (limited quality)
    print(f"Falling back to local generation for {img_path}...")
    result, _ = pipeline.generate(img_path)
    
    if isinstance(result, str) and os.path.exists(result):
        return result, "⚠️ Generated using local model (set TRIPO_API_KEY for pro quality)"
    
    # Ultimate fallback: placeholder cube
    b1 = trimesh.creation.box(extents=[1, 1, 1])
    mesh = trimesh.util.concatenate([b1])
    out_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    mesh.export(out_path)
    
    return out_path, "⚠️ Placeholder mode: Set TRIPO_API_KEY for real results"

# Professional Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Dhaatu: Professional Image-to-3D Generator")
    gr.Markdown("**Powered by Tripo3D Cloud** - Generate production-ready 3D models from images.")
    gr.Markdown("*Set `TRIPO_API_KEY` secret for professional results, or use local fallback.*")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Image")
            run_btn = gr.Button("🚀 Generate 3D Model", variant="primary")
        
        with gr.Column(scale=2):
            status_out = gr.Markdown("Status: Ready")
            output_3d = gr.Model3D(label="3D Model Viewer", clear_color=(0.1,0.1,0.1,1))
            
    run_btn.click(
        fn=generate_3d, 
        inputs=input_img, 
        outputs=[output_3d, status_out]
    )

if __name__ == "__main__":
    demo.launch()
