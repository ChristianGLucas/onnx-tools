from gen.messages_pb2 import GraphStructureRequest, OnnxModel
from nodes.fixtures import build_chain_model, build_tiny_mlp_model
from nodes.get_graph_structure import get_graph_structure
from nodes.testkit import assert_error, assert_ok, ax


def test_mlp_graph_is_gemm_then_relu():
    req = GraphStructureRequest(model=OnnxModel(model_data=build_tiny_mlp_model()))
    result = get_graph_structure(ax(), req)
    assert_ok(result)
    assert result.total_node_count == 2
    assert not result.truncated
    assert [n.op_type for n in result.nodes] == ["Gemm", "Relu"]
    assert list(result.nodes[0].inputs) == ["X", "W", "B"]
    assert list(result.nodes[0].outputs) == ["Z"]
    assert list(result.nodes[1].inputs) == ["Z"]
    assert list(result.nodes[1].outputs) == ["Y"]
    assert list(result.graph_inputs) == ["X"]
    assert list(result.graph_outputs) == ["Y"]


def test_max_nodes_caps_and_reports_truncation():
    req = GraphStructureRequest(model=OnnxModel(model_data=build_chain_model(10)), max_nodes=3)
    result = get_graph_structure(ax(), req)
    assert_ok(result)
    assert result.total_node_count == 10
    assert len(result.nodes) == 3
    assert result.truncated is True


def test_zero_max_nodes_uses_default_and_no_truncation_for_small_graph():
    req = GraphStructureRequest(model=OnnxModel(model_data=build_chain_model(5)), max_nodes=0)
    result = get_graph_structure(ax(), req)
    assert_ok(result)
    assert len(result.nodes) == 5
    assert result.truncated is False


def test_negative_max_nodes_is_invalid_argument():
    req = GraphStructureRequest(model=OnnxModel(model_data=build_tiny_mlp_model()), max_nodes=-1)
    result = get_graph_structure(ax(), req)
    assert_error(result, "INVALID_ARGUMENT")


def test_malformed_model_is_malformed_model_error():
    req = GraphStructureRequest(model=OnnxModel(model_data=b"garbage"))
    result = get_graph_structure(ax(), req)
    assert_error(result, "MALFORMED_MODEL")
