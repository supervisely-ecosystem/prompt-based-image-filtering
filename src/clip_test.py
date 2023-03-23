import os
import io
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import clip
from pathlib import Path
from PIL import Image, ImageOps

import supervisely as sly
from dotenv import load_dotenv
try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict

from src.clip_api import *

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
dataset_id = 61268
output_project_id = 18618
prompts = ["a photo of a cucumber"]
weights = [1.0]
model_zoo = clip.available_models()  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
model_name = "ViT-L/14@336px"

api = sly.Api()

model, preprocess = build_model(model_name, device)

input_prompts = preprocess_prompts(prompts, device)

img_infos = api.image.get_list(dataset_id)
img_ids = [img_info.id for img_info in img_infos]
print(len(img_ids))

all_logits = []

for img_ids_batch in tqdm(sly.batched(img_ids, batch_size)):
    imgs_bytes = api.image.download_bytes(dataset_id, img_ids_batch)
    imgs_pil = [load_image(io.BytesIO(img)) for img in imgs_bytes]
    imgs = [preprocess_image(load_image(io.BytesIO(img)), preprocess) for img in imgs_bytes]
    input_iamges = collate_batch(imgs, device)
    logits = infer_batch(model, input_iamges, input_prompts)
    all_logits.append(logits)

logits = collect_inference(all_logits)  # shape: [IMG, TEXT]
scores = calculate_scores(logits, weights)
scores = scores.flatten()

assert len(scores) == len(img_infos)

i_sort = np.argsort(scores)[::-1]

# Uploading copies
output_dataset_id = api.dataset.create(output_project_id, name=prompts[0], change_name_if_conflict=True).id
counter = 0
uploaded_ids = []
for i_batch in tqdm(sly.batched(i_sort, 100)):
    names, hashes = [], []
    for i_global in i_batch:
        names += [f"{counter:04}_{scores[i_global]:.4f}_{img_infos[i_global].name}"]
        hashes += [img_infos[i_global].hash]
        counter += 1
    uploaded_infos = api.image.upload_hashes(output_dataset_id, names, hashes)
    uploaded_ids += [info.id for info in uploaded_infos]

# Copy annotations
src_ids_sorted = np.array(img_ids)[i_sort].tolist()
api.annotation.copy_batch_by_ids(src_ids_sorted, uploaded_ids)
