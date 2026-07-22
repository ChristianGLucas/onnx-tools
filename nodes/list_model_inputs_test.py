from gen.messages_pb2 import OnnxModel
from nodes.fixtures import (
    build_add_model,
    build_model_with_initializer_as_input,
    build_tiny_mlp_model,
)
from nodes.list_model_inputs import list_model_inputs
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_model_input_has_dynamic_batch_axis():
    result = list_model_inputs(ax(), OnnxModel(model_data=build_tiny_mlp_model()))
    assert_ok(result)
    assert len(result.specs) == 1
    spec = result.specs[0]
    assert spec.name == "X"
    assert spec.elem_type == "FLOAT"
    assert list(spec.dims) == ["batch", "3"]


def test_add_model_inputs_are_two_fixed_dim_vectors():
    result = list_model_inputs(ax(), OnnxModel(model_data=build_add_model()))
    assert_ok(result)
    names = sorted(s.name for s in result.specs)
    assert names == ["A", "B"]
    for spec in result.specs:
        assert list(spec.dims) == ["3"]
        assert spec.elem_type == "FLOAT"


def test_initializer_redundantly_listed_as_input_is_excluded():
    result = list_model_inputs(ax(), OnnxModel(model_data=build_model_with_initializer_as_input()))
    assert_ok(result)
    names = [s.name for s in result.specs]
    assert names == ["X"], f"initializer W should be excluded from true inputs, got {names}"


def test_malformed_model_is_malformed_model_error():
    result = list_model_inputs(ax(), OnnxModel(model_data=b"garbage"))
    assert_error(result, "MALFORMED_MODEL")
