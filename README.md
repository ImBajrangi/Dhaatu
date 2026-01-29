---
title: Dhaatu
emoji: 🗿
colorFrom: purple
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Your Own Image-to-3D Generator - 100% Free
---

# 🗿 Dhaatu: Your Own Image-to-3D Generator

**100% Free • Runs on CPU • You Own It Completely**

This is YOUR custom 3D mesh generator that uses depth estimation to create 3D models from single images.

## Features
- **No GPU Required**: Works on free CPU-only Hugging Face Spaces
- **Depth-Based 3D**: Uses Intel's DPT model for accurate depth estimation
- **GLB Export**: Download 3D models ready for any 3D software
- **Adjustable Settings**: Control depth scale and resolution

## How It Works
1. Upload any image
2. The model estimates depth (what's close vs far)
3. Depth is converted to a 3D mesh with colors from your image
4. Download and use in Blender, Unity, or any 3D tool

## Credits
- Depth estimation: [Intel DPT](https://huggingface.co/Intel/dpt-hybrid-midas)
- Built with: Transformers, Trimesh, Gradio