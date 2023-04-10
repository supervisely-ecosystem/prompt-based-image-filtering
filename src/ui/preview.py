import os

import supervisely as sly

from random import randint

from supervisely.app.widgets import (
    Slider,
    LinePlot,
    Table,
    Image,
    Field,
    Card,
    Container,
)

import src.globals as g
import src.ui.output as output

# Preparing plot for data and hiding it until the inference is done.
plot = LinePlot("Images scores by prompt", show_legend=True, decimals_in_float=4)

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


def update_plot(xaxis, yaxis, text_prompt):
    global plot

    sly.logger.debug(f"Starting to draw plot with {len(xaxis)} images.")

    plot.loading = True

    plot.add_series(text_prompt, xaxis, yaxis)

    plot.loading = False

    sly.logger.debug(f"Plot with {len(xaxis)} images is drawn.")


def build_table(image_infos, scores):
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


def create_row(image_info, score):
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
    if datapoint.button_name != g.SELECT_BUTTON:
        return

    selected_image_id = datapoint.row[g.TABLE_COLUMNS[0]]
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
    selected_image_path = os.path.join(g.STATIC_DIR, selected_image_info.name)

    g.api.image.download(selected_image_id, selected_image_path)

    sly.logger.debug(
        f"Successfully downloaded image with id {selected_image_id} as {selected_image_path}."
    )

    image_preview.set(
        url=os.path.join("static", selected_image_info.name),
    )

    sly.logger.debug(f"Updated image preview with image from {selected_image_path}.")
