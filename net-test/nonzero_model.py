import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nonzero(x)


model = Model()
model.eval()

input_tensor = torch.tensor(
    [
        [0.0, 1.2, 0.0],
        [0.0, 0.0, 0.8],
        [2.5, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

torch.onnx.export(
    model,
    input_tensor,
    "nonzero.onnx",
    input_names=["input_tensor"],
    output_names=["nonzero_indices"],
    dynamic_axes={
        "input_tensor": {0: "rows", 1: "cols"},
        "nonzero_indices": {0: "num_nonzero"},
    },
)
