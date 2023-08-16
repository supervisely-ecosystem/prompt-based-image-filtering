import os
import numpy as np
import torch
import urllib
from PIL import Image, ImageOps
import open_clip

import supervisely as sly

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache")
sly.fs.mkdir(CACHE_DIR)
sly.logger.info(f"Models cache dir: {CACHE_DIR}")


def build_model(model_name, pretrained, device, model_data, jit=False):
    model_url = model_data.get("url")
    model_filename = model_data.get("path")
    model_path = os.path.join(CACHE_DIR, model_filename)

    if not os.path.exists(model_path):
        sly.logger.info(f"Model file wasn't found in cache in path {model_path}")
        sly.logger.info(f"Model {model_name} will be downloaded from {model_url}")

        download_model(model_url, model_path)

    sly.logger.info("Preparing the model...")

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
    return logits_per_image


def collect_inference(logits):
    return torch.cat(logits, 0).cpu().numpy()


def calculate_scores(logits: np.ndarray, weights: list):
    scores = (logits * weights).sum(-1)
    return scores


def download_model(source_url, dst_path):
    from src.ui.inference import inference_progress

    sly.logger.info(f"Download started from: {source_url}")
    sly.fs.mkdir(os.path.dirname(dst_path))

    with urllib.request.urlopen(source_url) as source, open(dst_path, "wb") as output:
        with inference_progress(
            message="Downloading model...",
            total=int(source.headers.get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as pbar:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                pbar.update(len(buffer))

    sly.logger.info(f"Download finished. Model saved to: {dst_path}")
