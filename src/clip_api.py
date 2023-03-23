import clip
import numpy as np
import torch
from PIL import Image, ImageOps


def get_models():
    return clip.available_models()

def build_model(model_name, device, jit=False):
    model, preprocess = clip.load(model_name, device=device, jit=jit)
    return model, preprocess

def load_image(image_path):    
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image

def preprocess_image(image_pil, preprocess) -> torch.Tensor:
    input_image = preprocess(image_pil)
    return input_image

def preprocess_prompts(prompts, device) -> torch.Tensor:
    input_prompts = clip.tokenize(prompts).to(device)
    return input_prompts

def collate_batch(input_images: list, device) -> torch.Tensor:
    input_images = torch.stack(input_images).to(device)
    return input_images

def infer_batch(model, input_images, input_promts):
    with torch.no_grad():
        logits_per_image, logits_per_text = model(input_images, input_promts)
    return logits_per_image  # [IMG, TEXT]

def collect_inference(logits):
    return torch.cat(logits, 0).cpu().numpy()

def calculate_scores(logits: np.ndarray, weights: list):
    # logits: [IMG, TEXT]
    scores = (logits * weights).sum(-1)
    return scores