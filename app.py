import gradio as gr
import spaces
from gradio_litmodel3d import LitModel3D

import os
import shutil
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = '/tmp/dhaatu_sessions'
os.makedirs(TMP_DIR, exist_ok=True)

# Global pipeline (will be loaded lazily inside GPU function)
pipeline = None

def get_pipeline():
    """Lazy load the pipeline only when GPU is available."""
    global pipeline
    if pipeline is None:
        print("Loading TRELLIS pipeline...")
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
        try:
            pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
        except:
            pass
        print("Pipeline loaded successfully!")
    return pipeline



def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    

def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess the input image for 3D generation."""
    pipe = get_pipeline()
    processed_image = pipe.preprocess_image(image)
    return processed_image



def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    

def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


@spaces.GPU(duration=120)
def generate_and_extract_glb(
    image: Image.Image,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
) -> Tuple[dict, str, str, str]:
    """Generate 3D model from image and extract GLB file."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    # Get pipeline (lazy load on first request)
    pipe = get_pipeline()
    
    # Generate 3D model using TRELLIS pipeline
    outputs = pipe.run(

        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    
    # Render preview video
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    
    # Extract GLB with PBR textures
    gs = outputs['gaussian'][0]
    mesh = outputs['mesh'][0]
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    
    state = pack_state(gs, mesh)
    torch.cuda.empty_cache()
    return state, video_path, glb_path, glb_path


@spaces.GPU
def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """Extract Gaussian splatting file from the 3D model."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


# Build Gradio UI
with gr.Blocks(delete_cache=(600, 600), theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    ## Dhaatu: Professional Image-to-3D Generator
    **Powered by TRELLIS** - Generate production-ready 3D assets from single images.
    
    * Upload an image and click "Generate & Extract GLB" to create a 3D asset
    * Background is automatically removed if no alpha channel exists
    * GLB files include PBR textures and are ready for game engines
    """)
    
    with gr.Row():
        with gr.Column():
            image_prompt = gr.Image(label="Input Image", format="png", image_mode="RGBA", type="pil", height=300)
            
            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
            
            with gr.Accordion(label="GLB Export Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Mesh Simplification", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Resolution", value=1024, step=512)

            generate_btn = gr.Button("🚀 Generate & Extract GLB", variant="primary")
            extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)

        with gr.Column():
            video_output = gr.Video(label="3D Preview", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(label="GLB Viewer", exposure=10.0, height=300)
            
            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)
    
    output_buf = gr.State()

    # Example images
    with gr.Row():
        examples = gr.Examples(
            examples=[
                f'assets/example_image/{image}'
                for image in os.listdir("assets/example_image")
            ] if os.path.exists("assets/example_image") else [],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=16,
        )

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        generate_and_extract_glb,
        inputs=[image_prompt, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, mesh_simplify, texture_size],
        outputs=[output_buf, video_output, model_output, download_glb],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_gs_btn, download_glb],
    )

    video_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[extract_gs_btn, download_glb, download_gs],
    )
    
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[download_glb, download_gs],
    )


# Launch
if __name__ == "__main__":
    # Pipeline is loaded lazily on first request (inside GPU function)
    demo.launch()
