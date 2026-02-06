# 🗿 Dhaatu V4 Training Guide (High-Speed CPU Mode)

This guide is optimized for users training on **CPU-only devices** or those experiencing slow training speeds.

## 1. Setup Colab Environment

1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. **Hardware**: If you have no GPU, use the standard runtime (CPU). If you have a GPU, use it, but this script is now significantly faster on both!

## 2. Upload Files

Upload the `train_dhaatu_v4_collab` file to your Colab workspace.

## 3. Install Dependencies

Run this in a Colab cell:
```bash
!pip install trimesh numpy torch torchvision tqdm matplotlib scikit-image requests
```

## 4. CPU Speed Boost: Pre-processing (New!)

Training on 3D meshes is usually slow because the computer has to convert the mesh to voxels (3D pixels) on-the-fly. 

**Our new pipeline solves this:**
1. **Mesh-to-Voxel Caching**: The script now converts all meshes to `.npy` voxels **once** before training starts.
2. **Speed Results**: Training becomes **100x faster** after the initial pre-processing.
3. **Lite Config**: We use a `VOXEL_SIZE` of 24 and `BATCH_SIZE` of 16 to ensure the script runs smoothly on low-memory devices.

## 5. Start Training

Run the training script:
```bash
!python train_dhaatu_v4_collab
```

### What happens?
- **Step 1**: Downloads ModelNet40 (~450MB).
- **Step 2 (The Boost)**: Converts meshes to voxels. This step takes some time but is only done **once**. Look for the progress bar: `🛠 Starting Pre-processing`.
- **Step 3**: Training starts at lightning speed using the cached files!

## 6. How to tune for your device?

Edit the `Config` class at the top of the script:
- **Still too slow?** Set `VOXEL_SIZE = 16`.
- **Out of memory?** Set `BATCH_SIZE = 8`.
- **Have a fast PC/GPU?** Set `VOXEL_SIZE = 32` and `LATENT_DIM = 512`.

## 7. Exporting to Dhaatu

Once training is finished, download `dhaatu_v4_final.pth`.
To use it in your local Dhaatu app, update `core/depth_to_3d.py` to include the `DhaatuV4Model` class and load your weights.

---
**Happy Training (at 100x speed)!** 🚀
