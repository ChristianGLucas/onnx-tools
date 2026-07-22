from gen.messages_pb2 import OnnxModel
from nodes.fixtures import build_add_model, build_tiny_mlp_model
from nodes.list_model_outputs import list_model_outputs
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_model_output_shape_and_type():
    result = list_model_outputs(ax(), OnnxModel(model_data=build_tiny_mlp_model()))
    assert_ok(result)
    assert len(result.specs) == 1
    spec = result.specs[0]
    assert spec.name == "Y"
    assert spec.elem_type == "FLOAT"
    assert list(spec.dims) == ["batch", "2"]


def test_add_model_output_fixed_dim():
    result = list_model_outputs(ax(), OnnxModel(model_data=build_add_model()))
    assert_ok(result)
    assert len(result.specs) == 1
    assert result.specs[0].name == "Y"
    assert list(result.specs[0].dims) == ["3"]


def test_malformed_model_is_malformed_model_error():
    result = list_model_outputs(ax(), OnnxModel(model_data=b"garbage"))
    assert_error(result, "MALFORMED_MODEL")


def test_oversized_model_is_model_too_large():
    huge = b"\x00" * (3 * 1024 * 1024 + 1)
    result = list_model_outputs(ax(), OnnxModel(model_data=huge))
    assert_error(result, "MODEL_TOO_LARGE")
