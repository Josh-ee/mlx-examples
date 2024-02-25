# ----------------
# ui.py
# ----------------

import gradio as gr
from PIL import Image
import numpy as np
import mlx.core as mx
import mlx.extension 
from stable_diffusion import StableDiffusion

def generate_images(prompt, n_images=4, steps=50, cfg=7.5, negative_prompt="", n_rows=2):
    sd = StableDiffusion()

    # Generate the latent vectors using diffusion
    latents = sd.generate_latents(
        prompt,
        n_images=n_images,
        cfg_weight=cfg,
        num_steps=steps,
        negative_text=negative_prompt,
    )
    for x_t in latents:
        # mx.simplify(x_t)
        # mx.simplify(x_t)
        mx.eval(x_t)

    # Decode them into images
    decoded = []
    for i in range(0, n_images):
        decoded_img = sd.decode(x_t[i:i+1])
        mx.eval(decoded_img)
        decoded.append(decoded_img)

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(n_rows * H, B // n_rows * W, C)
    x = (x * 255).astype(mx.uint8)

    # Convert to PIL Image
    return Image.fromarray(x.__array__())

iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(minimum=1, maximum=10, step=1, value=4, label="Number of Images"),
        gr.Slider(minimum=20, maximum=100, step=1, value=50, label="Steps"),
        # I updated the max to 20 (source: https://getimg.ai/guides/interactive-guide-to-stable-diffusion-guidance-scale-parameter) 
        gr.Slider(minimum=0.0, maximum=20.0, step=0.1, value=7.5, label="CFG Weight"),
        # I updated default to value, due to syntax error
        gr.Textbox(value="", label="Negative Prompt"),
        gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of Rows")
    ],
    outputs="image",
    title="Stable Diffusion Image Generator",
    description="Generate images from a textual prompt using Stable Diffusion"
)

iface.launch()