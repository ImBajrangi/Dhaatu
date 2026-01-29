import gradio as gr
import torch
import numpy as np
import trimesh
from core.inference import TrellisInferencePipeline
import tempfile
import os

# Initialize the pipeline
# Note: In a real HF space, weights would be pre-downloaded or cached
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = TrellisInferencePipeline(device=device)

def generate_3d(image):
    if image is None:
        return None
    
    # Save input image to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    print(f"Generating 3D model for {img_path}...")
    
    # Run the pipeline
    # In our PoC, this returns a dummy tensor [1, 4, 32, 32, 32]
    voxels = pipeline.generate(img_path)
    
    # Convert voxel tensor to a mesh/point cloud for Gradio Model3D
    # For now, we'll create a simple box or sphere as a placeholder 
    # that represents the output, until real weights are integrated.
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    # Save to GLB for Gradio to display
    out_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    mesh.export(out_path)
    
    return out_path

# Define Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 3DGen AI: Image-to-3D Generation")
    gr.Markdown("Transform 2D images into high-quality 3D assets using Structured Latent Diffusion.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            run_btn = gr.Button("Generate 3D Model", variant="primary")
        
        with gr.Column():
            output_3d = gr.Model3D(label="3D Model Viewer")
            
    run_btn.click(fn=generate_3d, inputs=input_img, outputs=output_3d)

if __name__ == "__main__":
    demo.launch()
