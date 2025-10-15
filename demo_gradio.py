import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

import os
import glob
import argparse
import numpy as np
import cv2
import torch
import gradio as gr

from PIL import Image, ImageChops, ImageEnhance

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from rembg import remove

from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook

import DPT.util.io
from torchvision.transforms import Compose
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet


"""
Get ZeST Ready
"""
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"


TARGET_SIZE = 1024
# Set the main diffusion data type to float16 for speed and lower VRAM usage.
DTYPE = torch.float16 # <---- OPTIMIZATION 1: Use Half-Precision for Speed

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise RuntimeError(
        "CUDA GPU not detected. This SDXL + ControlNet pipeline requires a CUDA-capable GPU.\n"
        "If you do have a GPU, ensure the NVIDIA driver + CUDA runtime are installed and PyTorch is the CUDA build."
    )

torch.cuda.empty_cache()

# # load SDXL pipeline
# # Load ControlNet with the faster DTYPE
# controlnet = ControlNetModel.from_pretrained(
#     controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=DTYPE
# ).to(device)

# # Load the main pipeline with the faster DTYPE
# pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
#     base_model_path,
#     controlnet=controlnet,
#     use_safetensors=True,
#     torch_dtype=DTYPE, # <---- OPTIMIZATION 1: Applying Half-Precision here
#     add_watermarker=False,
# )



# pipe.enable_vae_tiling()
# pipe.enable_model_cpu_offload()

# # stability & memory helpers (additions)
# torch.backends.cuda.matmul.allow_tf32 = True
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_attention_slicing()
# pipe.enable_vae_slicing()
# pipe.enable_vae_tiling()
# pipe.vae.to(dtype=torch.float32)  # Keep VAE in FP32 to prevent NaNs

# # register IP-Adapter attention hook
# pipe.unet = register_cross_attention_hook(pipe.unet)
# ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# --- START OF REVISED SETUP ---

# 🛑 VAE FIX: Define the path for the FP16-stable VAE
vae_fix_path = "madebyollin/sdxl-vae-fp16-fix" 

# Load ControlNet with the faster DTYPE and keep it on the GPU
controlnet = ControlNetModel.from_pretrained(
    controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=DTYPE
).to(device)

# Load the fixed VAE model in half-precision (DTYPE)
vae = AutoencoderKL.from_pretrained(
    vae_fix_path, torch_dtype=DTYPE, use_safetensors=True
)

# Load the main pipeline, injecting the fixed VAE, and remove manual .to(device)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    vae=vae, # <--- 1. INJECT THE FIXED VAE HERE
    use_safetensors=True,
    torch_dtype=DTYPE,
    add_watermarker=False,
) # REMOVED the .to(device) call to enable proper offloading

# 2. Enable Offload to manage memory (moves VAE to CPU when not used)
pipe.enable_model_cpu_offload()

# stability & memory helpers (additions)
torch.backends.cuda.matmul.allow_tf32 = True
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
# The VAE is now stable in float16, so the manual pipe.vae.to(dtype=torch.float32) is removed.

# register IP-Adapter attention hook
pipe.unet = register_cross_attention_hook(pipe.unet)
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# --- END OF REVISED SETUP ---


"""
Get Depth Model Ready
"""
model_path = "DPT/weights/dpt_hybrid-midas-501f0c75.pt"
net_w = net_h = 384
model = DPTDepthModel(
    path=model_path,
    backbone="vitb_rn50_384",
    non_negative=True,
    enable_attention_hooks=False,
)
normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

transform = Compose(
    [
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ]
)

model.eval()  # DPT runs on CPU by default; that’s fine for depth


def greet(input_image, material_exemplar):
    """
    Compute depth map from input_image
    """
    img = np.array(input_image)
    img_input = transform({"image": img})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    depth_min = prediction.min()
    depth_max = prediction.max()
    bits = 2
    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        # FIX: use prediction.dtype (not depth.dtype)
        out = np.zeros(prediction.shape, dtype=prediction.dtype)

    out = (out / 256).astype("uint8")
    depth_map = Image.fromarray(out).resize((TARGET_SIZE, TARGET_SIZE)).convert("L")

    """
    Process foreground decolored image
    """
    rm_bg = remove(input_image)  # returns RGBA PIL
    target_mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert("L")
    invert_target_mask = ImageChops.invert(target_mask.convert("RGB"))
    gray_target_image = input_image.convert("L").convert("RGB")
    gray_target_image = ImageEnhance.Brightness(gray_target_image).enhance(1.0)
    grayscale_img = ImageChops.darker(gray_target_image, target_mask.convert("RGB"))
    img_black_mask = ImageChops.darker(input_image, invert_target_mask)
    grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    init_img = grayscale_init_img

    """
    Process material exemplar and resize all images
    """
    ip_image = material_exemplar.resize((TARGET_SIZE, TARGET_SIZE))
    init_img = init_img.resize((TARGET_SIZE, TARGET_SIZE))
    mask = target_mask.resize((TARGET_SIZE, TARGET_SIZE)).convert("L")

    num_samples = 1

    # Autocast using the same faster DTYPE we loaded the models with
    autocast_ctx = torch.autocast("cuda", dtype=DTYPE) if device == "cuda" else torch.no_grad() # <---- OPTIMIZATION 2: Match Autocast DTYPE
    with autocast_ctx:
        images = ip_model.generate(
            pil_image=ip_image,
            image=init_img,
            control_image=depth_map,
            mask_image=mask,
            controlnet_conditioning_scale=0.8,
            num_samples=num_samples,
            num_inference_steps=24,
            seed=42,
        )


    final_image = np.array(images[0]).astype(np.float32) / 255.0
    final_image = np.clip(final_image, 0.0, 1.0)
    final_image = Image.fromarray((final_image * 255).astype(np.uint8))

    print("Generation complete.")
    print("final_image:", final_image)

    # return images[0]
    return final_image


input_image = gr.Image(type="pil", label="Input image")
input_image2 = gr.Image(type="pil", label="Material exemplar")

demo = gr.Interface(
    fn=greet,
    inputs=[input_image, input_image2],
    title="ZeST: Zero-Shot Material Transfer from a Single Image",
    description="Upload two images — input image and material exemplar. ZeST extracts the material from the exemplar and casts it onto the input image following original lighting cues.",
    outputs=["image"],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(share=True)
