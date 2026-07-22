from gen.messages_pb2 import OnnxModel
from nodes.fixtures import build_add_model, build_tiny_mlp_model
from nodes.get_model_size import get_model_size
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_model_size_matches_known_parameter_count():
    model_bytes = build_tiny_mlp_model()
    result = get_model_size(ax(), OnnxModel(model_data=model_bytes))
    assert_ok(result)
    assert result.total_parameters == 8  # 2x3 W + 2 B, see list_initializers_test.py
    assert result.initializer_count == 2
    assert result.model_size_bytes == len(model_bytes)


def test_add_model_has_zero_parameters():
    model_bytes = build_add_model()
    result = get_model_size(ax(), OnnxModel(model_data=model_bytes))
    assert_ok(result)
    assert result.total_parameters == 0
    assert result.initializer_count == 0
    assert result.model_size_bytes == len(model_bytes)


def test_malformed_model_is_malformed_model_error():
    result = get_model_size(ax(), OnnxModel(model_data=b"garbage"))
    assert_error(result, "MALFORMED_MODEL")
