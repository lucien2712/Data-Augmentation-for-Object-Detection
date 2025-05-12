import torch
import gc
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import StableDiffusionPipeline, StableDiffusionGLIGENPipeline
from .config import MODEL_CONFIGS, PROMPT_CONFIGS

def setup_models(description_model_type, generation_model_type):
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    
    # 設置描述模型
    desc_config = MODEL_CONFIGS[description_model_type]
    processor = Blip2Processor.from_pretrained(desc_config["processor"])
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        desc_config["model"],
        device_map={"": 0},
        torch_dtype=torch.float16
    )
    
    # 設置生成模型
    gen_config = MODEL_CONFIGS[generation_model_type]
    if generation_model_type == "stable_diffusion":
        pipe = StableDiffusionPipeline.from_pretrained(
            gen_config["model"],
            torch_dtype=torch.float16
        ).to(device)
    elif generation_model_type == "gligen":
        pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            gen_config["model"],
            variant="fp16",
            torch_dtype=torch.float16
        ).to(device)
    else:
        raise ValueError(f"不支援的生成模型類型: {generation_model_type}")
    
    return processor, blip_model, pipe

def generate_description(image, processor, model, prompt_type):

    prompt = PROMPT_CONFIGS[prompt_type]

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        device="cuda", dtype=torch.float16
    )

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.2,
        num_beams=8,
        length_penalty=1.5,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True
    )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def clean_memory():
   
    gc.collect()
    torch.cuda.empty_cache() 