import os
import shutil

import torch

import supervisely as sly

from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api: sly.Api = sly.Api.from_env()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_BATCH_SIZE = 32 if DEVICE == "cuda" else 16
sly.logger.info(f"Chosen device: {DEVICE}, batch size: {MODEL_BATCH_SIZE}")

TEAM_ID = sly.io.env.team_id()
WORKSPACE_ID = sly.io.env.workspace_id()

PROJECT_ID = sly.io.env.project_id(raise_not_found=False)
DATASET_ID = sly.io.env.dataset_id(raise_not_found=False)

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
MODELS = [
    ("coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k", "-", "2.55 GB"),
    ("coca_ViT-L-14", "laion2B-s13B-b90k", "75.5%", "2.55 GB"),
    ("ViT-L-14", "openai", "75.5%", "0.933 GB"),
    ("ViT-L-14", "laion2b_s32b_b82k", "75.3%", "0.933 GB"),
    ("ViT-L-14-336", "openai", "-", "0.933 GB"),
    ("ViT-g-14", "laion2b_s34b_b88k", "78.5%", "5.47 GB"),
    ("ViT-bigG-14", "laion2b_s39b_b160k", "80.1%", "10.2 GB"),
    ("convnext_base_w", "laion2b_s13b_b82k_augreg", "71.5%", "0.718 GB"),
    ("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup", "76.9%", "1.41 GB"),
    # ("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"),  # available only in timm pre-release (Apr 2023)
]
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
SELECTED_PROJECT = None
SELECTED_DATASET = None
