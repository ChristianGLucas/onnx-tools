"""Security/robustness tests: non-standard-domain rejection and
malformed-input handling — the "input the caller controls drives something
the node never intended" lens, exercised directly rather than hoped for.
Size/memory/DoS limits are the Axiom platform's job, not this package's.
"""

import onnx
from onnx import TensorProto, helper

from gen.messages_pb2 import GraphStructureRequest, InferenceRequest, OnnxModel, Tensor, TopKInferenceRequest
from nodes.fixtures import (
    build_add_model,
    build_custom_domain_model,
    build_nested_custom_domain_model,
    build_nested_standard_ops_model,
)
from nodes.get_graph_structure import get_graph_structure
from nodes.get_model_metadata import get_model_metadata
from nodes.list_model_inputs import list_model_inputs
from nodes.list_operators import list_operators
from nodes.run_inference import run_inference
from nodes.run_inference_top_k import run_inference_top_k
from nodes.testkit import assert_error, ax


def test_run_inference_never_executes_a_custom_domain_op():
    """The core guard: a graph node in a non-standard opset domain is
    rejected outright, never handed to onnxruntime — this is what stands
    between this package and running caller-influenced custom-op code."""
    model = build_custom_domain_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[Tensor(name="X", dtype="float32", shape=[1], float_data=[1.0])],
    )
    assert_error(run_inference(ax(), req), "UNSUPPORTED_OP")


def test_run_inference_top_k_never_executes_a_custom_domain_op():
    model = build_custom_domain_model()
    req = TopKInferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[Tensor(name="X", dtype="float32", shape=[1], float_data=[1.0])],
        k=1,
    )
    assert_error(run_inference_top_k(ax(), req), "UNSUPPORTED_OP")


def test_run_inference_rejects_custom_domain_op_nested_inside_if_subgraph():
    """REGRESSION (review finding): a non-standard-domain node hidden inside
    an If/Loop/Scan subgraph must be rejected exactly like a top-level one —
    the domain guard has to walk the WHOLE reachable graph, not just
    model.graph.node, or a caller can smuggle a custom op past it."""
    model = build_nested_custom_domain_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="cond", dtype="bool", shape=[], bool_data=[True]),
            Tensor(name="X", dtype="float32", shape=[3], float_data=[1.0, 2.0, 3.0]),
        ],
    )
    assert_error(run_inference(ax(), req), "UNSUPPORTED_OP")


def test_run_inference_top_k_rejects_custom_domain_op_nested_inside_if_subgraph():
    model = build_nested_custom_domain_model()
    req = TopKInferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="cond", dtype="bool", shape=[], bool_data=[True]),
            Tensor(name="X", dtype="float32", shape=[3], float_data=[1.0, 2.0, 3.0]),
        ],
        k=1,
    )
    assert_error(run_inference_top_k(ax(), req), "UNSUPPORTED_OP")


def test_run_inference_allows_legitimate_nested_control_flow():
    """The fix must not over-reject: an If model whose subgraphs use only
    standard ops still runs normally, on both branches."""
    model = build_nested_standard_ops_model()
    req_then = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="cond", dtype="bool", shape=[], bool_data=[True]),
            Tensor(name="X", dtype="float32", shape=[3], float_data=[1.0, 2.0, 3.0]),
        ],
    )
    result_then = run_inference(ax(), req_then)
    assert not result_then.error.code, result_then.error.message
    assert list(result_then.outputs[0].float_data) == [1.0, 2.0, 3.0]  # then branch: Identity

    req_else = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="cond", dtype="bool", shape=[], bool_data=[False]),
            Tensor(name="X", dtype="float32", shape=[3], float_data=[1.0, 2.0, 3.0]),
        ],
    )
    result_else = run_inference(ax(), req_else)
    assert not result_else.error.code, result_else.error.message
    assert list(result_else.outputs[0].float_data) == [-1.0, -2.0, -3.0]  # else branch: Neg


def test_list_operators_counts_nodes_nested_inside_if_subgraph():
    """REGRESSION (review finding): ListOperators must inventory nodes
    inside If/Loop/Scan subgraphs too — the top-level graph here has ONE
    node (the If itself), but the true reachable node count is 3 (If +
    Identity + Neg across both branches)."""
    result = list_operators(ax(), OnnxModel(model_data=build_nested_standard_ops_model()))
    assert not result.error.code
    assert result.total_nodes == 3
    counts = {op.op_type: op.count for op in result.operators}
    assert counts == {"If": 1, "Identity": 1, "Neg": 1}


def test_ai_onnx_ml_domain_is_allowed_not_rejected():
    """The standard "ai.onnx.ml" domain (classical-ML ops) must NOT be
    swept up by the custom-domain guard — only genuinely non-standard
    domains are rejected. Binarizer is a real, registered ai.onnx.ml op
    (threshold classifier: output 1.0 where input > threshold, else 0.0),
    so this is both a domain-allow-list check AND a genuine inference,
    hand-computable: threshold=2.0 on [1.0, 2.5, 5.0] -> [0, 1, 1]."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    node = helper.make_node(
        "Binarizer", inputs=["X"], outputs=["Y"], name="bin1", domain="ai.onnx.ml", threshold=2.0
    )
    graph = helper.make_graph([node], "ml_domain_graph", [x], [y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 1)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    model_bytes = model.SerializeToString()

    req = InferenceRequest(
        model=OnnxModel(model_data=model_bytes),
        inputs=[Tensor(name="X", dtype="float32", shape=[3], float_data=[1.0, 2.5, 5.0])],
    )
    result = run_inference(ax(), req)
    assert not result.error.code, f"ai.onnx.ml domain should be allowed, got {result.error.code}"
    assert list(result.outputs[0].float_data) == [0.0, 1.0, 1.0]


def test_negative_shape_dim_is_rejected():
    model = build_add_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="A", dtype="float32", shape=[-1], float_data=[1, 2, 3]),
            Tensor(name="B", dtype="float32", shape=[3], float_data=[1, 2, 3]),
        ],
    )
    assert_error(run_inference(ax(), req), "INVALID_INPUT")


def test_unknown_dtype_string_is_rejected():
    model = build_add_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="A", dtype="complex128", shape=[3], float_data=[1, 2, 3]),
            Tensor(name="B", dtype="float32", shape=[3], float_data=[1, 2, 3]),
        ],
    )
    assert_error(run_inference(ax(), req), "INVALID_INPUT")


def test_empty_bytes_model_is_malformed_not_crash():
    req = InferenceRequest(model=OnnxModel(model_data=b""), inputs=[])
    assert_error(run_inference(ax(), req), "MALFORMED_MODEL")


def test_random_junk_bytes_never_crash_any_inspection_node():
    junk = bytes([i % 256 for i in range(5000)])
    for fn in (
        lambda: list_model_inputs(ax(), OnnxModel(model_data=junk)),
        lambda: get_model_metadata(ax(), OnnxModel(model_data=junk)),
        lambda: list_operators(ax(), OnnxModel(model_data=junk)),
        lambda: get_graph_structure(ax(), GraphStructureRequest(model=OnnxModel(model_data=junk))),
    ):
        result = fn()  # must not raise
        assert result.error.code == "MALFORMED_MODEL"
