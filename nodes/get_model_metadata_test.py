from gen.messages_pb2 import OnnxModel
from nodes.fixtures import build_add_model, build_tiny_mlp_model
from nodes.get_model_metadata import get_model_metadata
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_model_metadata_matches_fixture_values():
    result = get_model_metadata(ax(), OnnxModel(model_data=build_tiny_mlp_model()))
    assert_ok(result)
    assert result.producer_name == "onnx-tools-tests"
    assert result.producer_version == "2.0"
    assert result.ir_version == 8
    assert result.model_version == 3
    assert result.domain == "com.onnxtools.tests"
    assert result.graph_name == "tiny_mlp"
    assert result.doc_string == "fixture: tiny 2-class MLP head"
    assert len(result.opset_imports) == 1
    assert result.opset_imports[0].domain == ""
    assert result.opset_imports[0].version == 17


def test_add_model_metadata():
    result = get_model_metadata(ax(), OnnxModel(model_data=build_add_model()))
    assert_ok(result)
    assert result.graph_name == "add_graph"
    assert result.model_version == 1


def test_malformed_model_is_malformed_model_error():
    result = get_model_metadata(ax(), OnnxModel(model_data=b"garbage"))
    assert_error(result, "MALFORMED_MODEL")
