import os

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

SLY_APP_DATA_DIR = sly.app.get_data_dir()

STATIC_DIR = os.path.join(SLY_APP_DATA_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

MODELS_COLUMNS = ["name", "size", "params"]
MODELS = ["placeholder01", "placeholder02", "placeholder03", "placeholder04", "placeholder05"]

FILTER_METHODS = ["above threshold", "below threshold"]
SORT_METHODS = {
    "asc": "Ascending 🔼",
    "desc": "Descending 🔽",
}

SELECTED_TEAM = None
SELECTED_WORKSPACE = None
SELECTED_PROJECT = None
SELECTED_DATASET = None
