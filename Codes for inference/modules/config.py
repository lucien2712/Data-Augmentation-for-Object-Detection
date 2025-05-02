
# 批次處理大小
BATCH_SIZE = 256

# 模型配置
MODEL_CONFIGS = {
    "blip_2_7b": {
        "processor": "Salesforce/blip2-opt-2.7b",
        "model": "Salesforce/blip2-opt-2.7b",
    },
    "blip_6_7b": {
        "processor": "Salesforce/blip2-opt-6.7b",
        "model": "Salesforce/blip2-opt-6.7b",
    },
    "stable_diffusion": {
        "model": "stabilityai/stable-diffusion-2-1"
    },
    "gligen": {
        "model": "masterful/gligen-1-4-generation-text-box"
    }
}

# 提示詞配置
PROMPT_CONFIGS = {
    "prompt1": "Question: What do you see in this image? Answer:",
    "prompt2": "Question: Describe this workplace image with exact details about: workers (clothing, positions), tools, and spatial relationships between elements. Answer:"
}

# 圖像生成參數
IMAGE_GEN_PARAMS = {
    "num_inference_steps": 150,
    "guidance_scale": 12,
    "height": 512,
    "width": 512,
    "seed": 42
}

# 提示詞模板
PROMPT_TEMPLATE = """A highly detailed, professional photograph of a real workplace environment,
8k resolution, natural lighting, sharp focus: {prompt}"""

# 負面提示詞
NEGATIVE_PROMPT = """blurry, low quality, distorted, unrealistic,
cartoon, anime, sketchy, duplicate, double image, mutated, deformed"""

# 輸出目錄配置
OUTPUT_DIRS = {
    "no_layout_prompt1": "no_layout_prompt1",
    "no_layout_prompt1_model2": "no_layout_prompt1_model2",
    "no_layout_prompt2_model2": "no_layout_prompt2_model2",
    "prompt2_model2_with_layout": "prompt2_model2_with_layout"
}

# 模式配置
MODES = {
    "no_layout_prompt1": {
        "description_model": "blip_2_7b",
        "prompt_type": "prompt1",
        "generation_model": "stable_diffusion",
        "use_layout": False,
        "output_dir": OUTPUT_DIRS["no_layout_prompt1"]
    },
    "no_layout_prompt1_model2": {
        "description_model": "blip_2_7b",
        "prompt_type": "prompt2",
        "generation_model": "stable_diffusion",
        "use_layout": False,
        "output_dir": OUTPUT_DIRS["no_layout_prompt1_model2"]
    },
    "no_layout_prompt2_model2": {
        "description_model": "blip_6_7b",
        "prompt_type": "prompt2",
        "generation_model": "stable_diffusion",
        "use_layout": False,
        "output_dir": OUTPUT_DIRS["no_layout_prompt2_model2"]
    },
    "prompt2_model2_with_layout": {
        "description_model": "blip_6_7b",
        "prompt_type": "prompt2",
        "generation_model": "gligen",
        "use_layout": True,
        "output_dir": OUTPUT_DIRS["prompt2_model2_with_layout"]
    }
} 