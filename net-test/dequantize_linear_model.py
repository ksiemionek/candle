import onnx
from onnx import helper, TensorProto
import numpy as np

"""
PyTorch have different convention for DequantizeLinear 
- operations takes as input qtensor (quantized tensor) that stores tensor of dtype (e.g. uint8) data, scale, zero_point but whole qtensor is of type e.g. quint8
"""

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

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)
onnx.save(model, "dequant.onnx")

x = np.array([[128, 129, 127], [255, 0, 200]], dtype=np.uint8)
expected = (x.astype(np.int32) - 128) * 0.05

print("Input x:\n", x)
print("Expected y = (x - 128) * 0.05:\n", expected)


# to make sure data is written as raw - candle reads data from raw_data buffer
m = onnx.load("dequant.onnx")
for init in m.graph.initializer:
    print(
        f"{init.name}: dims={list(init.dims)}, raw_data={len(init.raw_data)}B, float_data={list(init.float_data)}, int32_data={list(init.int32_data)}, raw_data_bytes={init.raw_data.hex()}"
    )
