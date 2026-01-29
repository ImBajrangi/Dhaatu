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
    
    # Save input image to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    print(f"Generating 3D model for {img_path}...")
    
    # Run the real production pipeline
    result, pbr_maps = pipeline.generate(img_path)
    
    # If result is already a path (from Shap-E), return it
    if isinstance(result, str) and os.path.exists(result):
        status = "✅ Successfully generated 3D model using OpenAI Shap-E!"
        return result, status
    
    # Fallback mesh generation (if real model failed)
    b1 = trimesh.creation.box(extents=[1, 1, 1])
    mesh = trimesh.util.concatenate([b1])
    out_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    mesh.export(out_path)
    
    error_msg = pipeline.real_model_error if hasattr(pipeline, 'real_model_error') else "Weight download incomplete"
    status = f"⚠️ Fallback Mode: Generated placeholder box. (Reason: {error_msg})"
    return out_path, status

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 3DGen AI: Image-to-3D Generation")
    gr.Markdown("Transform 2D images into high-quality 3D assets using OpenAI Shap-E.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            run_btn = gr.Button("Generate 3D Model", variant="primary")
        
        with gr.Column():
            status_out = gr.Markdown("Status: Ready")
            output_3d = gr.Model3D(label="3D Model Viewer")
            
    run_btn.click(
        fn=generate_3d, 
        inputs=input_img, 
        outputs=[output_3d, status_out]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
