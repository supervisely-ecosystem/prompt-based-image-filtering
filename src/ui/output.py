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

sort_checkbox = Checkbox("Sort images")
filter_checkbox = Checkbox("Filter images")

# Field with information threshold value.
threshold_input = InputNumber(value=0)
threshold_input.disable()
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

    project_id = destination.get_selected_project_id()
    dataset_id = destination.get_selected_dataset_id()

    if not project_id:

        sly.logger.info("Project was not selected. Creating new project.")

        project_id = create_project(destination.get_project_name())
    if not dataset_id:

        sly.logger.info("Dataset was not selected. Creating new dataset.")

        dataset_id = create_dataset(project_id, destination.get_dataset_name())

    save_progress.show()
    save_button.text = "Saving..."

    # Replace with real data.
    pseudo_iters = 10

    sly.logger.info(f"Start saving {pseudo_iters} images.")

    with save_progress(message="Saving in process...", total=pseudo_iters) as pbar:
        for i in range(1, pseudo_iters + 1):
            pbar.update(1)
            time.sleep(1)

    sly.logger.info(f"Successfully saved {pseudo_iters} images.")

    save_button.text = "Save"

    result_message.text = f"Successfully saved {pseudo_iters} images."
    result_message.status = "success"
    result_message.show()

    project_info = g.api.project.get_info_by_id(project_id)
    dataset_info = g.api.dataset.get_info_by_id(dataset_id)

    result_dataset.set(project_info, dataset_info)
    result_dataset.show()


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
