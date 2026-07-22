"""ONNX model fixtures built programmatically (via onnx.helper), NOT
committed binaries, so every test's ground truth is visible in the diff and
independently hand-computable — the independent-oracle discipline this
package's tests rely on.

Every fixture here is a tiny, fully-known graph: the weight/bias values are
chosen so a human can hand-compute the expected output with pen and paper,
independent of onnxruntime's own implementation.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_add_model() -> bytes:
    """Y = A + B, both length-3 float32 vectors. The simplest possible
    independent oracle: elementwise addition, hand-computable trivially."""
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    node = helper.make_node("Add", inputs=["A", "B"], outputs=["Y"], name="add1")
    graph = helper.make_graph([node], "add_graph", [a, b], [y])
    model = helper.make_model(
        graph,
        producer_name="onnx-tools-tests",
        producer_version="1.0",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    model.model_version = 1
    model.doc_string = "fixture: elementwise add"
    onnx.checker.check_model(model)
    return model.SerializeToString()


# Known weights for the tiny MLP fixture below — kept as module constants so
# tests can hand-compute expected outputs against the SAME numbers used to
# build the graph, without re-deriving them.
MLP_W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # (out=2, in=3)
MLP_B = np.array([-10.0, 1.0], dtype=np.float32)  # (out=2,) — negative on class 0 to exercise Relu clipping


def build_tiny_mlp_model() -> bytes:
    """Y = Relu(X @ W^T + B) — a 2-node graph (Gemm, Relu), one dynamic-axis
    graph input ("batch" x 3), one graph output (batch x 2), two
    initializers (W: 2x3, B: 2 -> 8 total parameters). Exercises graph
    structure, initializers, dynamic-axis input inspection, metadata, and
    inference (including Relu's clipping, since MLP_B[0] is negative) all
    from one fixture.

    Hand-computable ground truth for X=[1,1,1]:
      gemm = X @ W^T + B = [1+2+3-10, 4+5+6+1] = [-4, 16]
      relu = [max(-4,0), max(16,0)] = [0, 16]
    For X=[2,0,1]:
      gemm = [2*1+0*2+1*3-10, 2*4+0*5+1*6+1] = [-5, 15]
      relu = [0, 15]
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 2])
    w_init = numpy_helper.from_array(MLP_W, name="W")
    b_init = numpy_helper.from_array(MLP_B, name="B")
    gemm_node = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Z"], name="gemm1", transB=1)
    relu_node = helper.make_node("Relu", inputs=["Z"], outputs=["Y"], name="relu1")
    graph = helper.make_graph(
        [gemm_node, relu_node],
        "tiny_mlp",
        [x],
        [y],
        initializer=[w_init, b_init],
        doc_string="fixture: tiny 2-class MLP head",
    )
    model = helper.make_model(
        graph,
        producer_name="onnx-tools-tests",
        producer_version="2.0",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    model.model_version = 3
    model.domain = "com.onnxtools.tests"
    model.doc_string = "fixture: tiny 2-class MLP head"  # ModelProto's own doc_string, distinct from graph's
    onnx.checker.check_model(model)
    return model.SerializeToString()


def build_custom_domain_model() -> bytes:
    """A structurally-valid-enough (parseable) ONNX model whose single node
    lives in a non-standard opset domain — used to prove RunInference/
    RunInferenceTopK reject it as UNSUPPORTED_OP before ever attempting to
    execute it (never onnx.checker'd: the point is a domain neither this
    package nor onnx's checker has a registered schema for)."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    node = helper.make_node(
        "MyCustomOp", inputs=["X"], outputs=["Y"], name="custom1", domain="my.custom.domain"
    )
    graph = helper.make_graph([node], "custom_graph", [x], [y])
    model = helper.make_model(
        graph,
        producer_name="onnx-tools-tests",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("my.custom.domain", 1)],
    )
    model.ir_version = 8
    return model.SerializeToString()


def build_structurally_invalid_model() -> bytes:
    """A model that PARSES fine (valid protobuf bytes) but is structurally
    invalid ONNX: its one node references an input tensor ("NotDeclared")
    that is neither a graph input, an initializer, nor any earlier node's
    output. onnx.checker.check_model rejects this with a topological-sort
    error. Deliberately NOT checker-validated here — this is the fixture
    FOR testing that ValidateModel's checker call catches it."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    node = helper.make_node("Add", inputs=["X", "NotDeclared"], outputs=["Y"], name="add1")
    graph = helper.make_graph([node], "bad_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()


def build_model_with_initializer_as_input() -> bytes:
    """Some exporters redundantly list an initializer as a graph input too
    (an older, still-legal ONNX convention: an initializer with a matching
    graph.input entry is an "optional" input). Used to prove
    ListModelInputs/GetSchemaContract correctly EXCLUDE it — a caller
    neither needs to nor usefully can supply it, since it already has a
    baked-in value."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    w = helper.make_tensor_value_info("W", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    w_init = numpy_helper.from_array(np.array([1.0, 1.0], dtype=np.float32), name="W")
    node = helper.make_node("Add", inputs=["X", "W"], outputs=["Y"], name="add1")
    graph = helper.make_graph([node], "init_as_input_graph", [x, w], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def build_chain_model(n: int) -> bytes:
    """n sequential Identity nodes, X -> t0 -> t1 -> ... -> Y. Used to test
    GetGraphStructure's max_nodes cap/truncation on a graph with a known,
    exact node count."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    nodes = []
    prev = "X"
    for i in range(n):
        out_name = "Y" if i == n - 1 else f"t{i}"
        nodes.append(helper.make_node("Identity", inputs=[prev], outputs=[out_name], name=f"id{i}"))
        prev = out_name
    graph = helper.make_graph(nodes, "chain_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def build_int_classifier_model() -> bytes:
    """ArgMax-free int64-output identity-ish model: Y = X (a Cast node),
    used to exercise int64 tensor round-tripping distinct from the float
    fixtures above. X and Y are both length-4 int64 vectors."""
    x = helper.make_tensor_value_info("X", TensorProto.INT64, [4])
    y = helper.make_tensor_value_info("Y", TensorProto.INT64, [4])
    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"], name="identity1")
    graph = helper.make_graph([node], "int_identity_graph", [x], [y])
    model = helper.make_model(
        graph,
        producer_name="onnx-tools-tests",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()
