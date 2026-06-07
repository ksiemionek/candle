import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

"""
PyTorch have different convention for DequantizeLinear 
- operations takes as input qtensor (quantized tensor) that stores tensor of dtype (e.g. uint8) data, scale, zero_point but whole qtensor is of type e.g. quint8

Tests below are similar to those from onnx-docs implemented in candle eval.rs - https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html
- test for linear version is a bit modified, input+output for tests for per-axis and blocked versions are copied from docs
"""

######## DequantizeLinear - linear version ########

node = helper.make_node("DequantizeLinear", inputs=["x", "scale", "zero_point"], outputs=["y"])

""" Tests for scale and zero_point as scalar values and as tensors/arrays of dim=1 with 1 scalar value - onnx protobuf model reads scalars into one-el dim=1 tensors"""
# scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, dims=[], vals=np.float32(0.05).tobytes(), raw=True)
# zero_point_tensor = helper.make_tensor("zero_point", TensorProto.UINT8, dims=[], vals=np.uint8(128).tobytes(), raw=True)

scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, dims=[], vals=np.array([0.05], dtype=np.float32).tobytes(), raw=True)
zero_point_tensor = helper.make_tensor("zero_point", TensorProto.UINT8, dims=[], vals=np.array([128], dtype=np.uint8).tobytes(), raw=True)

# x: uint8, y: float32
graph_input = helper.make_tensor_value_info("x", TensorProto.UINT8, [None, None])
graph_output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None])

graph = helper.make_graph([node], "dequant_test", [graph_input], [graph_output], initializer=[scale_tensor, zero_point_tensor])

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])
onnx.checker.check_model(model)
onnx.save(model, "dequant.onnx")

x = np.array([[128, 129, 127], [255, 0, 200]], dtype=np.uint8)
expected = (x.astype(np.int32) - 128) * 0.05

print("Linear input x:\n", x)
print("Linear expected y = (x - 128) * 0.05:\n", expected)


# to make sure data is written as raw - candle reads data from raw_data buffer
m = onnx.load("dequant.onnx")
for init in m.graph.initializer:
    print(
        f"{init.name}: dims={list(init.dims)}, raw_data={len(init.raw_data)}B, float_data={list(init.float_data)}, int32_data={list(init.int32_data)}, raw_data_bytes={init.raw_data.hex()}"
    )

######## DequantizeLinear - per-axis version ########

node_axis = helper.make_node(
    "DequantizeLinear",
    inputs=["x", "scale", "zero_point"],
    outputs=["y"],
)

x = np.array(
    [
        [
            [[3, 89], [34, 200], [74, 59]],
            [[5, 24], [24, 87], [32, 13]],
            [[245, 99], [4, 142], [121, 102]],
        ],
    ],
    dtype=np.uint8,
)
x_scale = np.array([2, 4, 5], dtype=np.float32)
x_zero_point = np.array([84, 24, 196], dtype=np.uint8)
y = (x.astype(np.float32) - x_zero_point.reshape(1, 3, 1, 1).astype(np.float32)) * x_scale.reshape(1, 3, 1, 1)

graph_axis = helper.make_graph(
    [node_axis],
    "dequant_per_axis",
    [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3, 3, 2])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 3, 2])],
    initializer=[
        numpy_helper.from_array(x_scale, name="scale"),
        numpy_helper.from_array(x_zero_point, name="zero_point"),
    ],
)
model_axis = helper.make_model(graph_axis, opset_imports=[helper.make_opsetid("", 24)])
onnx.checker.check_model(model_axis)
onnx.save(model_axis, "dequant_per_axis.onnx")

print("Per-axis input x:\n", x)
print("\nPer-axis expected:\n", y)


######## DequantizeLinear - blocked version ########

node_blocked = helper.make_node(
    "DequantizeLinear",
    inputs=["x", "scale", "zero_point"],
    outputs=["y"],
    axis=1,
    block_size=2,
)

x = np.array(
    [
        [
            [[3, 89], [34, 200], [74, 59]],
            [[5, 24], [24, 87], [32, 13]],
            [[5, 12], [12, 33], [65, 42]],
            [[245, 99], [4, 142], [121, 102]],
        ],
    ],
    dtype=np.uint8,
)

x_scale = np.array(
    [
        [
            [[3.0, 2.0], [4.0, 1.0], [2.0, 2.0]],
            [[5.0, 2.0], [4.0, 3.0], [5.0, 2.0]],
        ],
    ],
    dtype=np.float32,
)
x_zero_point = np.array(
    [
        [
            [[1, 0], [0, 1], [2, 20]],
            [[3, 2], [4, 3], [15, 2]],
        ],
    ],
    dtype=np.uint8,
)

assert x_scale.shape == x_zero_point.shape
block_axis = 1
assert all(x.shape[i] == x_scale.shape[i] for i in range(len(x.shape)) if i != block_axis)
assert x.shape[block_axis] % x_scale.shape[block_axis] == 0
repeats = x.shape[block_axis] // x_scale.shape[block_axis]

x_scale_elementwise = np.repeat(x_scale, repeats=repeats, axis=block_axis)
x_zero_point_elementwise = np.repeat(x_zero_point, repeats=repeats, axis=block_axis)

y = (x.astype(np.float32) - x_zero_point_elementwise.astype(np.float32)) * x_scale_elementwise


graph_blocked = helper.make_graph(
    [node_blocked],
    "dequant_blocked",
    [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 4, 3, 2])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 3, 2])],
    initializer=[
        numpy_helper.from_array(x_scale, name="scale"),
        numpy_helper.from_array(x_zero_point, name="zero_point"),
    ],
)
model_blocked = helper.make_model(graph_blocked, opset_imports=[helper.make_opsetid("", 24)])
onnx.checker.check_model(model_blocked)
onnx.save(model_blocked, "dequant_blocked.onnx")

print("Blocked input x:\n", x)
print("\nBlocked expected:\n", y)
