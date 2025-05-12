import os
import torch
from PIL import Image
from .config import PROMPT_TEMPLATE, NEGATIVE_PROMPT, IMAGE_GEN_PARAMS

def generate_image(pipe, prompt, use_layout=False, boxes=None, phrases=None):
   
    enhanced_prompt = PROMPT_TEMPLATE.format(prompt=prompt)
    negative_prompt = NEGATIVE_PROMPT
    
    # 設置生成參數
    generator = torch.Generator(device="cuda").manual_seed(IMAGE_GEN_PARAMS["seed"])
    
    if use_layout:
        # 使用GLIGEN生成帶佈局的圖像
        if boxes is None or phrases is None:
            raise ValueError("使用 GLIGEN 時必須提供 boxes 和 phrases 參數")
        
        images = pipe(
            prompt=enhanced_prompt,
            gligen_phrases=phrases,
            gligen_boxes=boxes,
            num_inference_steps=IMAGE_GEN_PARAMS["num_inference_steps"],
            guidance_scale=IMAGE_GEN_PARAMS["guidance_scale"],
            height=IMAGE_GEN_PARAMS["height"],
            width=IMAGE_GEN_PARAMS["width"],
            gligen_scheduled_sampling_beta=0.9,
            negative_prompt=negative_prompt,
            generator=generator
        ).images
    else:
        # 使用標準Stable Diffusion生成圖像
        images = pipe(
            prompt=enhanced_prompt,
            num_inference_steps=IMAGE_GEN_PARAMS["num_inference_steps"],
            guidance_scale=IMAGE_GEN_PARAMS["guidance_scale"],
            height=IMAGE_GEN_PARAMS["height"],
            width=IMAGE_GEN_PARAMS["width"],
            negative_prompt=negative_prompt,
            generator=generator
        ).images
    
    return images[0]

def save_image(image, output_path):
   
  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path) 