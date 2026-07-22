from gen.messages_pb2 import OnnxModel
from nodes.fixtures import build_tiny_mlp_model
from nodes.get_schema_contract import get_schema_contract
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_model_schema_contract_is_the_union_of_inputs_outputs_opset():
    result = get_schema_contract(ax(), OnnxModel(model_data=build_tiny_mlp_model()))
    assert_ok(result)
    assert result.graph_name == "tiny_mlp"
    assert result.opset_version == 17
    assert len(result.inputs) == 1
    assert result.inputs[0].name == "X"
    assert list(result.inputs[0].dims) == ["batch", "3"]
    assert len(result.outputs) == 1
    assert result.outputs[0].name == "Y"
    assert list(result.outputs[0].dims) == ["batch", "2"]


def test_malformed_model_is_malformed_model_error():
    result = get_schema_contract(ax(), OnnxModel(model_data=b"garbage"))
    assert_error(result, "MALFORMED_MODEL")
