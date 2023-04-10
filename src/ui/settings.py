from supervisely.app.widgets import Card, RadioTable, Checkbox, InputNumber, Field, Container

import src.globals as g

# Building rows for RadioTable.
rows = []
for model in g.MODELS:
    row = [model[0], model[1]]
    rows.append(row)

# Field with model selection.
model_radio_table = RadioTable(columns=g.MODELS_COLUMNS, rows=rows)
model_radio_field = Field(
    title="Model",
    description="Choose one of the models from the list.",
    content=model_radio_table,
)

# Field with batch size input.
batch_size_input = InputNumber(value=32, min=1, max=1024)
batch_size_field = Field(
    title="Batch size",
    description="Choose the batch size in range from 1 to 1024.",
    content=batch_size_input,
)

# JIT field.
jit_checkbox = Checkbox(content="Enable JIT", checked=True)
jit_field = Field(
    title="Enable Just-In-Time compilation",
    description="JIT can speed up training, but may require more memory and increase runtime overhead.",
    content=jit_checkbox,
)

# Main card for all settings in the module.
card = Card(
    title="2️⃣ Settings",
    description="Choose the model and necessary settings for it.",
    content=Container(widgets=[model_radio_field, batch_size_field, jit_field]),
    lock_message="Select the dataset on step 1️⃣.",
)
card.lock()
