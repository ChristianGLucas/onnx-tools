from gen.messages_pb2 import OnnxModel
from nodes.fixtures import build_add_model, build_structurally_invalid_model, build_tiny_mlp_model
from nodes.testkit import assert_error, ax
from nodes.validate_model import validate_model


def test_well_formed_model_is_valid():
    result = validate_model(ax(), OnnxModel(model_data=build_tiny_mlp_model()))
    assert result.valid is True
    assert list(result.errors) == []
    assert not result.error.code


def test_another_well_formed_model_is_valid():
    result = validate_model(ax(), OnnxModel(model_data=build_add_model()))
    assert result.valid is True


def test_structurally_invalid_model_is_reported_invalid_not_crashed():
    result = validate_model(ax(), OnnxModel(model_data=build_structurally_invalid_model()))
    assert result.valid is False
    assert len(result.errors) >= 1
    assert "NotDeclared" in result.errors[0] or "topolog" in result.errors[0].lower()
    assert not result.error.code  # this is the node doing its job, not a processing failure


def test_garbage_bytes_is_reported_invalid_not_crashed():
    result = validate_model(ax(), OnnxModel(model_data=b"this is not protobuf at all \x00\x01\x02"))
    assert result.valid is False
    assert len(result.errors) >= 1
    assert not result.error.code


def test_empty_model_is_own_processing_error_not_valid_field():
    result = validate_model(ax(), OnnxModel(model_data=b""))
    assert_error(result, "MALFORMED_MODEL")
    assert result.valid is False  # default/unset, not a meaningful claim here
