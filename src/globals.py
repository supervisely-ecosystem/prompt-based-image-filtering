import os
import shutil

import torch

import supervisely as sly

from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_BATCH_SIZE = 32 if DEVICE == "cuda" else 16
sly.logger.info(f"Chosen device: {DEVICE}, batch size: {MODEL_BATCH_SIZE}")

TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()

PROJECT_ID = sly.env.project_id(raise_not_found=False)
DATASET_ID = sly.env.dataset_id(raise_not_found=False)

# Image table columns.
SELECT_BUTTON = "SELECT"
TABLE_COLUMNS = [
    "IMAGE ID",
    "FILE NAME",
    "WIDTH (PIXELS)",
    "HEIGHT (PIXELS)",
    "CONFIDENCE",
    SELECT_BUTTON,
]

SLY_APP_DATA_DIR = sly.app.get_data_dir()

# Define and create static directory.
ABSOLUTE_PATH = os.path.dirname(__file__)
STATIC_DIR = os.path.join(SLY_APP_DATA_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Batch size for uploading images to the dataset.
BATCH_SIZE = 100

# Define and copy placeholder image for Image preview widget to static directory.
PLACEHOLDER = "placeholder.png"
dst_file = os.path.join(STATIC_DIR, PLACEHOLDER)
shutil.copy(PLACEHOLDER, dst_file)

# Columns for RadioTable widget with models.
MODELS_COLUMNS = ["Name", "Pretrained on", "Top-1 accuracy on ImageNet", "Size"]
# List of available models.
MODELS = {
    ("ViT-L-14", "openai", "75.5%", "0.933 GB"): {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "path": "clip/ViT-L-14.pt",
    },
    ("coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k", "-", "2.55 GB"): {
        "url": "https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/blobs/f22c34acef2b7a5d1ed28982a21077de651363eaaebcf34a3f10676e17837cb8",
    },
    ("coca_ViT-L-14", "laion2B-s13B-b90k", "75.5%", "2.55 GB"): {
        "url": "https://huggingface.co/laion/CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CoCa-ViT-L-14-laion2B-s13B-b90k/blobs/73725652298ad76ed2162caffdae96d8653a05d7a29b6281103e4df81d0ff8ea",
    },
    ("ViT-L-14", "laion2b_s32b_b82k", "75.3%", "0.933 GB"): {
        "url": "https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/blobs/5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b",
    },
    ("ViT-L-14-336", "openai", "-", "0.933 GB"): {
        "url": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        "path": "clip/ViT-L-14-336px.pt",
    },
    ("ViT-g-14", "laion2b_s34b_b88k", "78.5%", "5.47 GB"): {
        "url": "https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-ViT-g-14-laion2B-s34B-b88K/blobs/9ef136f407986fb607cd37a823eba38a3b6f95e8ec702b3d1687252985d84750",
    },
    ("ViT-bigG-14", "laion2b_s39b_b160k", "80.1%", "10.2 GB"): {
        "url": "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/blobs/0d5318839ad03607c48055c45897c655a14c0276a79f6b867934ddd073760e39",
    },
    ("convnext_base_w", "laion2b_s13b_b82k_augreg", "71.5%", "0.718 GB"): {
        "url": "https://huggingface.co/laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/blobs/249e2302c1670bb04476792196f788ff046fedef61191a24983e61b6eca56987",
    },
    ("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup", "76.9%", "1.41 GB"): {
        "url": "https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blobs/4572137af44b2e26f01f638337a59688ec289e9363e15c08dde16640afb86988",
    },
}
# Prompt weights.
WEIGHTS = [1.0]

# Available methods for filtering and sorting images.
FILTER_METHODS = ["above threshold", "below threshold"]
SORT_METHODS = {"desc": "Descending 🔽", "asc": "Ascending 🔼"}


class State:
    def __init__(self):
        self.text_prompt = None
        self.image_infos = None
        self.scores = None
        self.i_sort = None
        self.continue_inference = True

    def get_params(self):
        return self.image_infos, self.scores, self.i_sort


STATE = State()

SELECTED_TEAM = None
SELECTED_WORKSPACE = None
PROJECT_META = None
SELECTED_PROJECT = None
SELECTED_DATASET = None
