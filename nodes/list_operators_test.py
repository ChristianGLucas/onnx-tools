from gen.messages_pb2 import OnnxModel
from nodes.fixtures import build_add_model, build_tiny_mlp_model
from nodes.list_operators import list_operators
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_model_has_gemm_and_relu_once_each():
    result = list_operators(ax(), OnnxModel(model_data=build_tiny_mlp_model()))
    assert_ok(result)
    assert result.total_nodes == 2
    counts = {op.op_type: op.count for op in result.operators}
    assert counts == {"Gemm": 1, "Relu": 1}


def test_add_model_has_single_add_op():
    result = list_operators(ax(), OnnxModel(model_data=build_add_model()))
    assert_ok(result)
    assert result.total_nodes == 1
    counts = {op.op_type: op.count for op in result.operators}
    assert counts == {"Add": 1}


def test_malformed_model_is_malformed_model_error():
    result = list_operators(ax(), OnnxModel(model_data=b"garbage"))
    assert_error(result, "MALFORMED_MODEL")
