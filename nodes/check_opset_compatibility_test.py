from gen.messages_pb2 import OnnxModel, OpsetCompatRequest
from nodes.check_opset_compatibility import check_opset_compatibility
from nodes.fixtures import build_tiny_mlp_model
from nodes.testkit import assert_error, assert_ok, ax


def test_model_opset_17_compatible_with_target_18():
    req = OpsetCompatRequest(model=OnnxModel(model_data=build_tiny_mlp_model()), target_opset=18)
    result = check_opset_compatibility(ax(), req)
    assert_ok(result)
    assert result.compatible is True
    assert result.model_opset == 17
    assert result.target_opset == 18


def test_model_opset_17_compatible_with_target_17_equal():
    req = OpsetCompatRequest(model=OnnxModel(model_data=build_tiny_mlp_model()), target_opset=17)
    result = check_opset_compatibility(ax(), req)
    assert_ok(result)
    assert result.compatible is True


def test_model_opset_17_incompatible_with_older_target_10():
    req = OpsetCompatRequest(model=OnnxModel(model_data=build_tiny_mlp_model()), target_opset=10)
    result = check_opset_compatibility(ax(), req)
    assert_ok(result)
    assert result.compatible is False
    assert result.model_opset == 17
    assert "newer opset" in result.note


def test_zero_target_opset_is_invalid_argument():
    req = OpsetCompatRequest(model=OnnxModel(model_data=build_tiny_mlp_model()), target_opset=0)
    result = check_opset_compatibility(ax(), req)
    assert_error(result, "INVALID_ARGUMENT")


def test_malformed_model_is_malformed_model_error():
    req = OpsetCompatRequest(model=OnnxModel(model_data=b"garbage"), target_opset=17)
    result = check_opset_compatibility(ax(), req)
    assert_error(result, "MALFORMED_MODEL")
