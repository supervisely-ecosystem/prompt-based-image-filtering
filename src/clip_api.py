import numpy as np
import torch
from PIL import Image, ImageOps
import open_clip

# from src.model_zoo import model_zoo


# def get_models():
#    return model_zoo


def build_model(model_name, pretrained, device, jit=False):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained, device=device, jit=jit
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def load_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image


def preprocess_image(image_pil, preprocess) -> torch.Tensor:
    input_image = preprocess(image_pil)
    return input_image


def preprocess_prompts(prompts, tokenizer, device) -> torch.Tensor:
    input_prompts = tokenizer(prompts).to(device)
    return input_prompts


def collate_batch(input_images: list, device) -> torch.Tensor:
    input_images = torch.stack(input_images).to(device)
    return input_images


def infer_batch(model: open_clip.CLIP, input_images, input_prompts):
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(input_images)
        text_features = model.encode_text(input_prompts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits_per_image = image_features @ text_features.T
    return logits_per_image  # [IMG, TEXT]


def collect_inference(logits):
    return torch.cat(logits, 0).cpu().numpy()


def calculate_scores(logits: np.ndarray, weights: list):
    # logits: [IMG, TEXT]
    scores = (logits * weights).sum(-1)
    return scores
