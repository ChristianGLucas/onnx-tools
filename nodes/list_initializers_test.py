from gen.messages_pb2 import OnnxModel
from nodes.fixtures import build_add_model, build_tiny_mlp_model
from nodes.list_initializers import list_initializers
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_model_initializers_match_known_shapes():
    result = list_initializers(ax(), OnnxModel(model_data=build_tiny_mlp_model()))
    assert_ok(result)
    assert result.total_initializer_count == 2
    # W is (2,3)=6 elements, B is (2,)=2 elements -> 8 total parameters.
    assert result.total_parameters == 8
    by_name = {i.name: i for i in result.initializers}
    assert set(by_name) == {"W", "B"}
    assert list(by_name["W"].shape) == [2, 3]
    assert by_name["W"].element_count == 6
    assert list(by_name["B"].shape) == [2]
    assert by_name["B"].element_count == 2
    assert by_name["W"].elem_type == "FLOAT"


def test_add_model_has_no_initializers():
    result = list_initializers(ax(), OnnxModel(model_data=build_add_model()))
    assert_ok(result)
    assert result.total_initializer_count == 0
    assert result.total_parameters == 0
    assert list(result.initializers) == []


def test_malformed_model_is_malformed_model_error():
    result = list_initializers(ax(), OnnxModel(model_data=b"garbage"))
    assert_error(result, "MALFORMED_MODEL")
