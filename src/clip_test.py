import os
import io
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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

from src import tags

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

dataset_id = 61265  # 2k
# dataset_id = 61175  # 100
output_project_id = 18615  # 2k
# output_project_id = 18619  # 100

prompts = ["a cucumber"]
weights = [1.0]

"ViT-B-32", "openai"
"ViT-L-14", "openai"
"ViT-L-14", "laion2b_s32b_b82k"
"ViT-L-14-336", "openai"
"ViT-g-14", "laion2b_s12b_b42k"
"coca_ViT-L-14", "laion2B-s13B-b90k"
"coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k"

model_name, pretrained = "coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k"

api = sly.Api()

model, preprocess, tokenizer = build_model(model_name, pretrained, device)

input_prompts = preprocess_prompts(prompts, tokenizer, device)

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
dataset_name = f"{prompts[0]} {model_name} {pretrained}"
output_dataset_id = api.dataset.create(
    output_project_id, name=dataset_name, change_name_if_conflict=True
).id
counter = 0
uploaded_ids = []
for i_batch in tqdm(sly.batched(i_sort, 100)):
    names, hashes, metas = [], [], []
    for i_global in i_batch:
        names += [f"{counter:04}_{scores[i_global]:.4f}_{img_infos[i_global].name}"]
        metas += [img_infos[i_global].meta]
        hashes += [img_infos[i_global].hash]
        counter += 1
    uploaded_infos = api.image.upload_hashes(output_dataset_id, names, hashes, metas=metas)
    uploaded_ids += [info.id for info in uploaded_infos]

# Copy annotations
src_ids_sorted = np.array(img_ids)[i_sort].tolist()
api.annotation.copy_batch_by_ids(src_ids_sorted, uploaded_ids)

# Add tags clip_score
tag_name = "clip_score"
tag_meta = sly.TagMeta(tag_name, "any_number")
_, tag_meta = tags.create_tag_meta(output_project_id, tag_meta)
for score, img_id in tqdm(zip(scores[i_sort], uploaded_ids)):
    tags.add_img_tag(img_id, tag_meta, score)
