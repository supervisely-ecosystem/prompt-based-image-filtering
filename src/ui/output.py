import time

from collections import namedtuple
from datetime import datetime

import supervisely as sly
from supervisely.app.widgets import (
    Checkbox,
    Card,
    Container,
    Field,
    InputNumber,
    RadioGroup,
    DestinationProject,
    Button,
    Progress,
    Text,
    DatasetThumbnail,
)

import src.globals as g
import src.ui.inference as inference

sort_checkbox = Checkbox("Sort images")
filter_checkbox = Checkbox("Filter images")

# Field with information threshold value.
threshold_input = InputNumber(value=0.25, min=0.0, max=1.0, step=0.001)
threshold_field = Field(
    title="Threshold",
    description="Current threshold value for the images.",
    content=threshold_input,
)

# Field with filter method selection.
filter_method_radio = RadioGroup(
    items=[RadioGroup.Item(value=method, label=method.capitalize()) for method in g.FILTER_METHODS]
)
filter_method_field = Field(
    title="Keep images",
    description="Choose which images to keep: above or below the threshold.",
    content=filter_method_radio,
)

# Container with all setings for filter method.
filter_containter = Container(widgets=[threshold_field, filter_method_field])
filter_containter.hide()

# Field with sort method selection.
sort_method_radio = RadioGroup(
    items=[RadioGroup.Item(value=method, label=label) for method, label in g.SORT_METHODS.items()]
)
sort_method_field = Field(
    title="Sort images by",
    description="Choose the sorting method for the images.",
    content=sort_method_radio,
)
sort_method_field.hide()

# Message when no method was selected.
no_method_message = Text("At least one method should be selected.", status="error")
no_method_message.hide()

# Field with all method widgets.
method_field = Field(
    title="Output method",
    description="Choose the output method for the images.",
    content=Container(
        widgets=[
            filter_checkbox,
            filter_containter,
            sort_checkbox,
            sort_method_field,
        ]
    ),
)

destination = DestinationProject(g.SELECTED_WORKSPACE, project_type="images")

add_confidence_checkbox = Checkbox("Add confidence tag to the image metadata", checked=True)

save_button = Button("Save")

save_progress = Progress()
save_progress.hide()

result_message = Text()

result_dataset = DatasetThumbnail()
result_dataset.hide()

save_container = Container(
    widgets=[
        no_method_message,
        destination,
        add_confidence_checkbox,
        save_button,
        save_progress,
        result_message,
        result_dataset,
    ]
)

card = Card(
    title="5️⃣ Output",
    description="Choose the output for the images and save the results.",
    content=Container(widgets=[method_field, save_container]),
    lock_message="Complete the inference on step 3️⃣.",
)
card.lock()


@sort_checkbox.value_changed
def sort_method(is_checked):
    if is_checked:
        sort_method_field.show()
    else:
        sort_method_field.hide()


@filter_checkbox.value_changed
def filter_method(is_checked):
    if is_checked:
        filter_containter.show()
    else:
        filter_containter.hide()


@save_button.click
def save():
    # Check if at least one method was selected.
    if not sort_checkbox.is_checked() and not filter_checkbox.is_checked():
        no_method_message.show()

        sly.logger.debug("Save button clicked, but no method was selected.")

        return

    no_method_message.hide()
    result_dataset.hide()
    result_message.hide()

    Filter = namedtuple("Filter", ["active", "method", "threshold"])
    Sort = namedtuple("Sort", ["active", "method"])

    filter_settings = Filter(
        active=filter_checkbox.is_checked(),
        method=filter_method_radio.get_value(),
        threshold=threshold_input.get_value(),
    )

    sort_settings = Sort(active=sort_checkbox.is_checked(), method=sort_method_radio.get_value())

    add_tag = add_confidence_checkbox.is_checked()

    sly.logger.debug(
        f"Save button was clicked. Filter settings: {filter_settings}. "
        f"Sort settings: {sort_settings}. Add tag: {add_tag}."
    )

    image_infos, scores, i_sort = inference.image_infos, inference.scores, inference.i_sort

    if sort_settings.active:
        sly.logger.debug(f"Sorting is active, the sorting method is {sort_settings.method}.")

        if sort_settings.method == "desc":
            image_infos = [image_infos[i] for i in i_sort]
            scores = [scores[i] for i in i_sort]
        elif sort_settings.method == "asc":
            image_infos = [image_infos[i] for i in i_sort[::-1]]
            scores = [scores[i] for i in i_sort[::-1]]

        sly.logger.debug("Images were sorted along with their scores.")

    if filter_settings.active:
        sly.logger.debug(
            f"Filtering is active, the filtering method is {filter_settings.method} "
            f"with threshold {filter_settings.threshold}."
        )

        sly.logger.debug(f"Starting to filter {len(image_infos)} images.")

        if filter_settings.method == "above threshold":
            image_infos = [
                image_infos[i]
                for i, score in enumerate(scores)
                if score >= filter_settings.threshold
            ]
        elif filter_settings.method == "below threshold":
            image_infos = [
                image_infos[i]
                for i, score in enumerate(scores)
                if score <= filter_settings.threshold
            ]

        sly.logger.debug(f"Finished filtering. {len(image_infos)} images left.")

    project_id = destination.get_selected_project_id()
    dataset_id = destination.get_selected_dataset_id()
    if dataset_id == g.SELECTED_DATASET:
        sly.logger.warning("Same dataset was selected. Showing warning message and stopping.")
        sly.app.show_dialog(
            title="Same dataset was selected",
            description=(
                "It's not allowed to save results to the same dataset. "
                "Please select another dataset or create a new one."
            ),
            status="warning",
        )
        return

    if not project_id:
        sly.logger.info("Project was not selected. Creating new project.")

        project_id = create_project(destination.get_project_name())
    if not dataset_id:
        sly.logger.info("Dataset was not selected. Creating new dataset.")

        dataset_id = create_dataset(project_id, destination.get_dataset_name())

    sly.logger.info(f"Project ID: {project_id}. Dataset ID: {dataset_id}.")

    # project_meta = update_project_meta(project_id)

    image_ids = [image.id for image in image_infos]
    # sly.logger.debug(f"Created list with {len(image_ids)} image IDs, which will be uploaded.")

    # annotations = download_annotations(image_ids, project_meta)

    save_progress.show()
    save_button.text = "Saving..."

    sly.logger.info(f"Start saving {len(image_infos)} images.")

    with save_progress(message="Saving in process...", total=len(image_infos)) as pbar:
        uploaded_image_ids = []
        prefix = 0

        for batched_image_infos in sly.batched(image_infos, g.BATCH_SIZE):
            sly.logger.debug(f"Starting to upload batch of {len(batched_image_infos)} images.")

            ids = [image_info.id for image_info in batched_image_infos]
            metas = [image_info.meta for image_info in batched_image_infos]
            names = []

            for i in range(len(batched_image_infos)):
                old_name = batched_image_infos[i].name
                new_name = f"{str(prefix).zfill(5)}_{old_name}"
                names.append(new_name)
                prefix += 1

            uploaded_images = g.api.image.upload_ids(dataset_id, names, ids, metas=metas)
            uploaded_image_ids.extend([image.id for image in uploaded_images])

            sly.logger.debug(f"Successfully uploaded batch of {len(batched_image_infos)} images.")
            pbar.update(len(batched_image_infos))

    sly.logger.info(f"Finished uploading {len(uploaded_image_ids)} images.")

    g.api.annotation.copy_batch_by_ids(image_ids, uploaded_image_ids)
    sly.logger.info(f"Suceessfully copied annotations for {len(uploaded_image_ids)} images.")

    save_button.text = "Save"

    result_message.text = f"Successfully saved {len(uploaded_image_ids)} images."
    result_message.status = "success"
    result_message.show()

    project_info = g.api.project.get_info_by_id(project_id)
    dataset_info = g.api.dataset.get_info_by_id(dataset_id)

    result_dataset.set(project_info, dataset_info)
    result_dataset.show()


def update_project_meta(project_id):
    meta_json = g.api.project.get_meta(g.SELECTED_PROJECT)
    project_meta = sly.ProjectMeta.from_json(meta_json)

    sly.logger.debug(
        f"Successfully downloaded project meta for original project with ID {g.SELECTED_PROJECT}."
    )

    g.api.project.update_meta(project_id, project_meta)

    sly.logger.debug(f"Successfully updated project meta for new project with ID {project_id}.")

    return project_meta


def download_annotations(image_ids, project_meta):
    sly.logger.debug(f"Starting download of annotations for {len(image_ids)} images.")

    annotation_infos = g.api.annotation.download_batch(g.SELECTED_DATASET, image_ids)

    annotation_jsons = [annotation_info.annotation for annotation_info in annotation_infos]

    annotations = [sly.Annotation.from_json(json, project_meta) for json in annotation_jsons]

    sly.logger.debug(
        f"Downloaded {len(annotations)} annotations from dataset with id {g.SELECTED_DATASET}."
    )

    return annotations


def create_project(project_name):
    # If the name is not specified, use the search query as the name.
    if not project_name:
        from src.ui.inference import text_prompt

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        sly.logger.debug(
            f"Project name is not specified, using text prompt {text_prompt} and timestamp."
        )
        project_name = f"{timestamp}_({text_prompt})"

    project = g.api.project.create(g.WORKSPACE_ID, project_name, change_name_if_conflict=True)

    sly.logger.info(f"Project with name {project_name} and id {project.id} was created.")

    return project.id


def create_dataset(project_id, dataset_name):
    # If the name is not specified, use the search query as the name.
    if not dataset_name:
        from src.ui.inference import text_prompt

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        sly.logger.debug(
            f"Project name is not specified, using text prompt {text_prompt} and timestamp."
        )
        dataset_name = f"{timestamp}_({text_prompt})"

    dataset = g.api.dataset.create(project_id, dataset_name, change_name_if_conflict=True)

    sly.logger.info(f"Dataset with name {dataset_name} and id {dataset.id} was created.")

    return dataset.id
