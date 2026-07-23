from gen.messages_pb2 import GraphStructureRequest, OnnxModel
from nodes.fixtures import build_tiny_mlp_model
from nodes.get_graph_structure import get_graph_structure
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_graph_is_gemm_then_relu():
    req = GraphStructureRequest(model=OnnxModel(model_data=build_tiny_mlp_model()))
    result = get_graph_structure(ax(), req)
    assert_ok(result)
    assert [n.op_type for n in result.nodes] == ["Gemm", "Relu"]
    assert list(result.nodes[0].inputs) == ["X", "W", "B"]
    assert list(result.nodes[0].outputs) == ["Z"]
    assert list(result.nodes[1].inputs) == ["Z"]
    assert list(result.nodes[1].outputs) == ["Y"]
    assert list(result.graph_inputs) == ["X"]
    assert list(result.graph_outputs) == ["Y"]


def test_malformed_model_is_malformed_model_error():
    req = GraphStructureRequest(model=OnnxModel(model_data=b"garbage"))
    result = get_graph_structure(ax(), req)
    assert_error(result, "MALFORMED_MODEL")
