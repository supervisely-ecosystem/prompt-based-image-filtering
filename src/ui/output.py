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

method_field = Field(
    title="Output method",
    description="Choose the output method for the images.",
    content=Container(widgets=[sort_checkbox, filter_checkbox]),
)

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
    items=[RadioGroup.Item(value=method.capitalize()) for method in g.FILTER_METHODS]
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
    content=Container(widgets=[method_field, filter_containter, sort_method_field, save_container]),
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
