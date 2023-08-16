import supervisely as sly
from supervisely.app.widgets import (
    Card,
    SelectDataset,
    Button,
    Container,
    DatasetThumbnail,
    Text,
)

import src.globals as g
import src.ui.settings as settings
import src.ui.inference as inference
import src.ui.preview as preview
import src.ui.output as output

dataset_thumbnail = DatasetThumbnail()
dataset_thumbnail.hide()

load_button = Button("Load data")
change_dataset_button = Button("Change dataset", icon="zmdi zmdi-lock-open")
change_dataset_button.hide()

no_dataset_message = Text(
    "Please, select a dataset before clicking the button.",
    status="warning",
)
no_dataset_message.hide()


def unlock_cards():
    settings.card.unlock()
    inference.card.unlock()


if g.DATASET_ID and g.PROJECT_ID:
    # If the app was loaded from a dataset.
    sly.logger.debug("App was loaded from a dataset.")

    # Stting values to the widgets from environment variables.
    select_dataset = SelectDataset(default_id=g.DATASET_ID, project_id=g.PROJECT_ID)

    g.SELECTED_TEAM = g.TEAM_ID
    g.SELECTED_WORKSPACE = g.WORKSPACE_ID
    g.SELECTED_PROJECT = g.PROJECT_ID
    g.PROJECT_META = sly.ProjectMeta.from_json(g.api.project.get_meta(g.SELECTED_PROJECT))
    g.SELECTED_DATASET = g.DATASET_ID

    # Hiding unnecessary widgets.
    select_dataset.hide()
    load_button.hide()

    # Creating a dataset thumbnail to show.
    dataset_thumbnail.set(
        g.api.project.get_info_by_id(g.PROJECT_ID),
        g.api.dataset.get_info_by_id(g.DATASET_ID),
    )
    dataset_thumbnail.show()

    unlock_cards()

elif g.PROJECT_ID:
    # If the app was loaded from a project: showing the dataset selector in compact mode.
    sly.logger.debug("App was loaded from a project.")

    # g.SELECTED_TEAM = g.TEAM_ID
    # g.SELECTED_WORKSPACE = g.WORKSPACE_ID
    # g.SELECTED_PROJECT = g.PROJECT_ID

    select_dataset = SelectDataset(project_id=g.PROJECT_ID, compact=True, show_label=False)
else:
    # If the app was loaded from ecosystem: showing the dataset selector in full mode.
    sly.logger.debug("App was loaded from ecosystem.")

    select_dataset = SelectDataset()

# Inout card with all widgets.
card = Card(
    "1️⃣ Input dataset",
    "Images from the selected dataset will be loaded.",
    content=Container(
        widgets=[
            dataset_thumbnail,
            select_dataset,
            load_button,
            change_dataset_button,
            no_dataset_message,
        ]
    ),
    lock_message="Inference is running...",
)


@load_button.click
def load_dataset():
    """Handles the load button click event. Reading values from the SelectDataset widget,
    calling the API to get project, workspace and team ids (if they're not set),
    unlocking the settings card and showing the dataset thumbnail.
    """
    # Reading the dataset id from SelectDataset widget.
    dataset_id = select_dataset.get_selected_id()

    if not dataset_id:
        # If the dataset id is empty, showing the warning message.
        no_dataset_message.show()
        return

    # Hide the warning message if dataset was selected.
    no_dataset_message.hide()

    # Changing the values of the global variables to access them from other modules.
    g.SELECTED_DATASET = dataset_id

    # Disabling the dataset selector and the load button.
    select_dataset.disable()
    load_button.hide()

    # Showing the unlock button to change the dataset.
    change_dataset_button.show()

    sly.logger.debug(f"Calling API with dataset ID {dataset_id} to get project ID.")

    g.SELECTED_PROJECT = g.api.dataset.get_info_by_id(dataset_id).project_id
    g.PROJECT_META = sly.ProjectMeta.from_json(g.api.project.get_meta(g.SELECTED_PROJECT))
    g.SELECTED_WORKSPACE = g.api.project.get_info_by_id(g.SELECTED_PROJECT).workspace_id
    g.SELECTED_TEAM = g.api.workspace.get_info_by_id(g.SELECTED_WORKSPACE).team_id

    sly.logger.debug(
        f"Recived IDs from the API. Selected team: {g.SELECTED_TEAM}, "
        f"selected workspace: {g.SELECTED_WORKSPACE}, selected project: {g.SELECTED_PROJECT}"
    )

    dataset_thumbnail.set(
        g.api.project.get_info_by_id(g.SELECTED_PROJECT),
        g.api.dataset.get_info_by_id(g.SELECTED_DATASET),
    )

    dataset_thumbnail.show()

    unlock_cards()


@change_dataset_button.click
def unlock_input():
    """Handles the change dataset button click event. Hiding the dataset thumbnail,
    showing the dataset selector and the load button, locking the settings card."""
    select_dataset.enable()
    load_button.show()
    change_dataset_button.hide()

    settings.card._lock_message = "Select the dataset on step 1️⃣."
    settings.card.lock()

    inference.card.lock()
    preview.card.lock()
    # preview.plot.clean_up()  # Requires PR with new function in LinePlot.
    preview.table.hide()
    preview.image_preview.hide()
    output.card.lock()
