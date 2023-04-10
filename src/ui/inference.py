import time
import io
import os

import torch

import numpy as np
import supervisely as sly

from supervisely.app.widgets import Card, Input, Field, Container, Progress, Button, Flexbox, Text

import src.globals as g
import src.clip_api as clip_api
import src.ui.input as input
import src.ui.settings as settings
import src.ui.preview as preview
import src.ui.output as output

image_infos = None
scores = None
i_sort = None

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

start_inference_button = Button(text="Start inference")

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
            start_inference_button,
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

    global text_prompt
    text_prompt = text_prompt_input.get_value()
    if not text_prompt:
        text_prompt_message.show()

        sly.logger.debug("Start inference button was clicked, but no text prompt was entered.")

        return

    text_prompt_message.hide()
    preview.table.hide()
    preview.image_preview.hide()
    preview.card.lock()
    preview.rows.clear()
    preview.image_preview.clean_up()
    preview.image_preview.set(url=os.path.join("static", g.PLACEHOLDER))

    inference_progress.show()
    start_inference_button.text = "Preparing..."

    global continue_inference
    continue_inference = True

    model_name, pretrained = g.MODELS[settings.model_radio_table.get_selected_row_index()]
    bath_size = settings.batch_size_input.get_value()
    jit = settings.jit_checkbox.is_checked()

    sly.logger.info(
        f"Starting inference with model: {model_name}, batch size: {bath_size}, JIT: {jit}. Text prompt: {text_prompt}"
    )

    input.card.lock()
    settings.card._lock_message = "Inference is running..."
    settings.card.lock()
    output.card.lock()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sly.logger.info(f"Using device: {device}.")
    model, preprocess, tokenizer = clip_api.build_model(model_name, pretrained, device)
    sly.logger.info(
        f"Model was built. Name: {model_name}, pretrained: {pretrained}, batch size: {bath_size}, JIT: {jit}."
    )

    input_prompts = clip_api.preprocess_prompts([text_prompt], tokenizer, device)
    sly.logger.info(f"Input prompts were preprocessed. Text prompt: {text_prompt}.")

    global image_infos
    image_infos = g.api.image.get_list(g.SELECTED_DATASET)
    image_ids = [image_info.id for image_info in image_infos]
    sly.logger.info(
        f"Loaded {len(image_ids)} images from selected dataset with id {g.SELECTED_DATASET}."
    )

    with inference_progress(message="Inference is running...", total=len(image_ids)) as pbar:
        sly.logger.info(f"Starting inference loop with batch size: {bath_size}.")
        start_inference_button.text = "Running..."

        all_logits = []

        for batched_image_ids in sly.batched(image_ids, bath_size):
            batched_image_bytes = g.api.image.download_bytes(g.SELECTED_DATASET, batched_image_ids)

            sly.logger.debug(f"Downloaded {len(batched_image_bytes)} images as bytes.")

            images_pil = [
                clip_api.load_image(io.BytesIO(image_bytes)) for image_bytes in batched_image_bytes
            ]

            sly.logger.debug(f"Loaded {len(images_pil)} images as PIL.")
            images = [clip_api.preprocess_image(image_pil, preprocess) for image_pil in images_pil]
            sly.logger.debug(f"Preprocessed {len(images)} images.")

            input_images = clip_api.collate_batch(images, device)
            logits = clip_api.infer_batch(model, input_images, input_prompts)
            all_logits.append(logits)

            pbar.update(len(batched_image_ids))

    logits = clip_api.collect_inference(all_logits)

    global scores
    scores = clip_api.calculate_scores(logits, g.WEIGHTS).flatten()

    assert len(scores) == len(image_infos)

    global i_sort
    i_sort = np.argsort(scores)[::-1]

    inference_message.text = "Inference finished successfully."
    inference_message.status = "success"

    sly.logger.info(f"Inference finished successfully. Text prompt: {text_prompt}.")

    i_list, score_list = zip(*[(i, score) for i, score in enumerate(scores[i_sort[::-1]])])
    preview.update_plot(i_list, score_list, text_prompt)

    preview.build_table(image_infos, scores)

    preview.card.unlock()
    output.card.unlock()

    inference_message.show()
    input.card.unlock()
    settings.card.unlock()

    start_inference_button.text = "Start inference"
