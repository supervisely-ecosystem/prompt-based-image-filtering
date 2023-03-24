import time

import supervisely as sly

from supervisely.app.widgets import Card, Input, Field, Container, Progress, Button, Flexbox, Text

import src.globals as g
import src.ui.input as input
import src.ui.settings as settings
import src.ui.preview as preview
import src.ui.output as output

# Field with text prompt input for filtering.
text_prompt_input = Input(minlength=1, placeholder="Enter the text prompt here...")
text_prompt_field = Field(
    title="Text prompt for filtering",
    description="Enter the text prompt for filtering the images.",
    content=text_prompt_input,
)

# Message if no text prompt was entered.
text_prompt_message = Text(
    text="Please, enter the text prompt for filtering the images.", status="error"
)
text_prompt_message.hide()

# Buttons Flexbox.
start_inference_button = Button(text="Start inference")
stop_inference_button = Button(text="Stop inference", button_type="danger")
stop_inference_button.hide()
buttons_flexbox = Flexbox(widgets=[start_inference_button, stop_inference_button])

# Progress bar for inference.
inference_progress = Progress()
inference_progress.hide()

# Inference result message.
inference_message = Text()
inference_message.hide()

# Card with all module widgets.
card = Card(
    title="3️⃣ Inference",
    description="Make predictions using selected model with input data.",
    content=Container(
        widgets=[
            text_prompt_field,
            text_prompt_message,
            buttons_flexbox,
            inference_progress,
            inference_message,
        ]
    ),
    lock_message="Select the dataset on step 1️⃣.",
)
card.lock()


@start_inference_button.click
def start_inference():

    inference_message.hide()

    text_prompt = text_prompt_input.get_value()
    if not text_prompt:
        text_prompt_message.show()

        sly.logger.debug("Start inference button was clicked, but no text prompt was entered.")

        return

    text_prompt_message.hide()

    inference_progress.show()
    stop_inference_button.show()
    start_inference_button.text = "Running..."

    global continue_inference
    continue_inference = True

    model = g.MODELS[settings.model_radio_table.get_selected_row_index()]
    bath_size = settings.batch_size_input.get_value()
    jit = settings.jit_checkbox.is_checked()

    sly.logger.info(
        f"Starting inference with model: {model}, batch size: {bath_size}, JIT: {jit}. Text prompt: {text_prompt}"
    )

    input.card.lock()
    settings.card._lock_message = "Inference is running..."
    settings.card.lock()
    preview.card.lock()
    preview.gallery_container.hide()
    output.card.lock()

    # Replace with real data.
    pseudo_iters = 10

    with inference_progress(message="Inference is running...", total=pseudo_iters) as pbar:
        for i in range(1, pseudo_iters + 1):
            if not continue_inference:
                break
            pbar.update(1)
            time.sleep(2)

    if continue_inference:

        inference_message.text = "Inference finished successfully."
        inference_message.status = "success"

        preview.load_images()
        preview.card.unlock()
        preview.gallery_container.show()
        output.card.unlock()
    else:
        inference_message.text = "Inference was stopped."
        inference_message.status = "warning"

    inference_message.show()
    input.card.unlock()
    settings.card.unlock()

    start_inference_button.text = "Start inference"


@stop_inference_button.click
def stop_inference():
    global continue_inference
    continue_inference = False

    stop_inference_button.hide()
    start_inference_button.text = "Stopping..."

    sly.logger.debug("Stop inference button clicked. Trying to stop inference...")
