# Setup Guide

## 1. Start With The Project

- Clone the GitHub repo.
- Create a conda or virtual environment with Python 3.10.

```bash
conda create -n prproj_lvface
```

- Run:

```bash
pip install -r requirements.txt
```

## 2. Requirements.txt Notes

- Insightface cannot be built normally without Microsoft C++ Build Tools, so you need a prebuilt `.whl` file from a public repository.

### Install Insightface

Because it needs Microsoft C++ Build Tools to build locally, it will not build if you do not have them.

Go to: https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/773

Scroll down and you will see:

- Download the prebuilt Insightface package.
  - Based on your Python version, download the appropriate `.whl` file:
    - For Python 3.10
    - For Python 3.11
    - For Python 3.12

Choose Python 3.10 and download the wheel. Then go into the folder with the wheel and run:

```bash
pip install insightface-0.7.3-cp310-cp310-win_amd64.whl --no-deps
```

**Must** use `--no-deps` because this package will try to overwrite NumPy 1.23 and cause some code to not run.

Pip will tell you that some dependency of insightface is missing. Install the rest of the dependencies manually, following pip's message if needed:

```bash
pip install albumentations==1.3.1 cython==3.0.10 easydict==1.13 onnx==1.14.1 requests==2.31.0 numpy==1.23.5
```

Keep `numpy==1.23.5` at the end of the command to ensure NumPy does not get overwritten.

## 3. Configure CUDA

If you do not have CUDA and cuDNN installed:

### CUDA 12.4

Link: https://developer.nvidia.com/cuda-12-4-0-download-archive

- Choose the exe (local) option.
- Download and follow the instructions until "custom installation options".
- To avoid overwriting the current display driver, only tick CUDA and untick the other options.
- You do not need to care about Visual Studio integration; just proceed.

### cuDNN 9.2

If you have not downloaded cuDNN yet, or if it is not in the right folder, do the following:

1. Go to the NVIDIA cuDNN Archive and download cuDNN 9.x for CUDA 12.x. Download the Windows `.zip` file, not the `.exe` installer.
2. Unzip the downloaded file.
3. Inside, you will see folders named `bin`, `include`, and `lib`.
4. Copy the contents of those folders directly into your CUDA installation path:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
```

Make sure all the `cudnn*.dll` files from the downloaded `bin` folder are pasted into the CUDA `bin` folder, the files from `include` and `lib` of cuDNN are pasted into CUDA `include` and `lib` too.

## 4. Run `eval_ijbc.py`
Evaluate on the IJB-C (9GB) dataset. Download from: https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view
(or go to github: https://github.com/DanJun6737/TransFace and scroll down to download ijb-c)

Put the dataset in folder data/ 

Put the weights in models/, download them from https://huggingface.co/bytedance-research/LVFace/tree/main (need .pt files for eval_ijbc.py - evaluating on big datasets, .onnx for inference_onnx.py - calculate similarities between single pair of images)

Command (change paths to fit your setup)
```bash
python eval_ijbc.py --model-prefix models/LVFace-B_Glint360K_from_onnx.pt --image-path data/ijb-testsuite/ijb/IJBC/ --result-dir results --target IJBC --network vit_b_dp005_mask_005 > LVFace-B_Glint360K.log 2>&1 &
```

The run will create a log file outside and results in results/ folder.