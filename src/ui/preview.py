import os

from typing import List, Union

import supervisely as sly
from supervisely.app.widgets import (
    LinePlot,
    Table,
    Image,
    Card,
    Container,
)

import src.globals as g

# Preparing plot for data and hiding it until the inference is done.
plot = LinePlot("Images scores by prompt", show_legend=True, decimals_in_float=4)

# Preparing table for data and hiding it until the inference is done.
table = Table(fixed_cols=1, width="100%", per_page=15, sort_column_id=4, sort_direction="desc")
table.hide()
rows = []

image_preview = Image()
image_preview.set(url=os.path.join("static", g.PLACEHOLDER))
image_preview.hide()

# Card for all widgets in the module.
card = Card(
    title="4️⃣ Preview",
    description="Preview the results of the model.",
    content=Container(
        widgets=[plot, Container(widgets=[table, image_preview], direction="horizontal")]
    ),
    lock_message="Complete the inference on step 3️⃣",
)
card.lock()


def update_plot(xaxis: List[int], yaxis: List[float], text_prompt: str):
    """Updates the plot with new data, which contain number of images and their scores.
    Uses text_prompt to name the series on the plot and add it to legend.

    Args:
        xaxis (List[int]): list of image numbers.
        yaxis (List[float]): list of image scores (confidence).
        text_prompt (str): text prompt, used for inference. Will be used as a name for the series
            and to add it to legend.
    """
    global plot

    sly.logger.debug(f"Starting to draw plot with {len(xaxis)} images.")

    plot.loading = True

    plot.add_series(text_prompt, xaxis, yaxis)

    plot.loading = False

    sly.logger.debug(f"Plot with {len(xaxis)} images is drawn.")


def build_table(image_infos: List[sly.api.image_api.ImageInfo], scores: List[float]):
    """Builds the table with image data from dataset and their scores.

    Args:
        image_infos (List[sly.api.image_api.ImageInfo]): list of image infos from dataset (id, name, width, height).
        scores (List[float]): scores (confidence) for each image (in the same order as image_infos).
    """
    global table, rows

    table.loading = True
    sly.logger.debug(f"Starting to build table with {len(image_infos)} images.")

    for image, score in zip(image_infos, scores):
        rows.append(create_row(image, score))

    table_data = {"columns": g.TABLE_COLUMNS, "data": rows}

    table.read_json(table_data)

    sly.logger.debug(f"Table with {len(image_infos)} images is built.")

    table.loading = False

    table.show()
    image_preview.show()


def create_row(image_info: sly.api.image_api.ImageInfo, score: float) -> List[Union[int, str]]:
    """Creates a row for the table with image data and score, also adds a button to select the image.

    Args:
        image_info (sly.api.image_api.ImageInfo): image info from dataset (id, name, width, height).
        score (float): score (confidence) for the image.

    Returns:
        List[Union[int, str]]: list of values for the row in the table.
    """
    return [
        image_info.id,
        image_info.name,
        image_info.width,
        image_info.height,
        f"{score:.4f}",
        sly.app.widgets.Table.create_button(g.SELECT_BUTTON),
    ]


@table.click
def handle_table_button(datapoint: sly.app.widgets.Table.ClickedDataPoint):
    """Handles the click on the button in the table. Downloads the image and updates the image preview.

    Args:
        datapoint (sly.app.widgets.Table.ClickedDataPoint): data point with the button name and row data.
    """
    if datapoint.button_name != g.SELECT_BUTTON:
        return

    # Reading the image id from the table row.
    selected_image_id = datapoint.row[g.TABLE_COLUMNS[0]]
    # Getting the image info from the dataset by id.
    selected_image_info = g.api.image.get_info_by_id(selected_image_id)

    if not selected_image_info:
        # If there was en error while getting the image info, deleting the row with the image id
        # and show the error message for the user.

        sly.logger.error(f"Can't find image with id {selected_image_id} in the dataset.")
        sly.app.show_dialog(
            "Image not found",
            f"Can't find image with id {selected_image_id} in the dataset.",
            status="error",
        )

        table.delete_row(g.TABLE_COLUMNS[0], selected_image_id)

        sly.logger.debug(f"Deleted the row with id {selected_image_id} from the table.")

        return

    sly.logger.debug(
        f"Image with id {selected_image_id} was selected in the table. Image info retrieved successfully."
    )
    # Defining the path in static directory to download the image for the preview widget.
    selected_image_path = os.path.join(g.STATIC_DIR, selected_image_info.name)

    g.api.image.download(selected_image_id, selected_image_path)

    sly.logger.debug(
        f"Successfully downloaded image with id {selected_image_id} as {selected_image_path}."
    )

    # Updating the image preview widget with the downloaded image.
    image_preview.set(
        url=os.path.join("static", selected_image_info.name),
    )

    sly.logger.debug(f"Updated image preview with image from {selected_image_path}.")
