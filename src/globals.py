import os
import shutil

import supervisely as sly

from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api: sly.Api = sly.Api.from_env()

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
MODELS_COLUMNS = ["Name", "Pretrained"]
# List of available models.
MODELS = [
    ("ViT-L-14", "openai"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-L-14-336", "openai"),
    ("ViT-g-14", "laion2b_s12b_b42k"),
    ("coca_ViT-L-14", "laion2B-s13B-b90k"),
    ("coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k"),
]
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
