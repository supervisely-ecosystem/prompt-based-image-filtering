from random import randint

from supervisely.app.widgets import (
    Slider,
    LinePlot,
    GridGallery,
    Field,
    Card,
    Container,
)

import src.globals as g
import src.ui.output as output

# Preparing plot for data and hiding it until the inference is done.
plot = LinePlot("Placeholder")
plot.hide()

# Field with treshold slider.
threshold_slider = Slider(min=0, max=1, step=0.05, show_input=True, show_input_controls=True)
threshold_field = Field(
    title="Threshold",
    description="Choose the threshold for the model.",
    content=threshold_slider,
)
# Cards for the left and right galleries.
before_gallery = GridGallery(columns_number=5, show_opacity_slider=False)
after_gallery = GridGallery(columns_number=5, show_opacity_slider=False)
before_card = Card(
    title="Before",
    description="Images before the model.",
    content=before_gallery,
)
after_card = Card(
    title="After",
    description="Images after the model.",
    content=after_gallery,
)
# Container for both galleries.
gallery_container = Container(widgets=[before_card, after_card], direction="horizontal")
gallery_container.hide()

# Card for all widgets in the module.
card = Card(
    title="4️⃣ Preview",
    description="Preview the results of the model.",
    content=Container(widgets=[plot, threshold_field, gallery_container]),
    lock_message="Complete the inference on step 3️⃣",
)
card.lock()


def load_images():
    before_gallery.loading = True
    after_gallery.loading = True

    before_gallery.clean_up()
    after_gallery.clean_up()

    # Replace with a real data.
    pseudo_images = g.api.image.get_list(g.SELECTED_DATASET)
    for _ in range(10):
        images_number = len(pseudo_images)
        before_gallery.append(pseudo_images[randint(0, images_number - 1)].preview_url)
        after_gallery.append(pseudo_images[randint(0, images_number - 1)].preview_url)

    before_gallery.loading = False
    after_gallery.loading = False


def update_plot():
    # Replace with a real data.
    global plot

    plot.loading = True

    pseudo_xaxis = 20
    pseudo_x = [i for i in range(1, pseudo_xaxis + 1)]
    pseudo_y = [randint(-100, 100) for i in range(pseudo_xaxis)]
    plot.add_series("Series placeholder", pseudo_x, pseudo_y)

    plot.loading = False


@threshold_slider.value_changed
def treshold_changed(treshold):
    output.threshold_input.value = treshold
    load_images()
