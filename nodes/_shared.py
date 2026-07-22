"""Shared ONNX parsing / tensor-marshaling / inference helpers for
christiangeorgelucas/onnx-tools.

Every node funnels its model decoding and tensor conversion through the
helpers here so the security posture (size caps, standard-domain-only
inference, deterministic session settings) and the Tensor<->numpy convention
are defined exactly once, not once per node.

SECURITY / DETERMINISM, read this before touching create_session/run_session:
  - Every model is size-checked FIRST (MAX_MODEL_BYTES), before any parse or
    session-creation work — this bounds memory/CPU cost on the raw input
    before we ever touch it.
  - onnxruntime telemetry is disabled at import (disable_telemetry_events()).
  - Every inference session is CPU-only (providers=["CPUExecutionProvider"]
    explicitly — never left to onnxruntime's own provider auto-selection),
    single-threaded (intra_op_num_threads=inter_op_num_threads=1), and
    sequential (ExecutionMode.ORT_SEQUENTIAL) — this makes a fixed model +
    fixed input deterministic (parallel reduction order is a real source of
    nondeterminism onnxruntime otherwise permits).
  - Before a session is ever created, the graph is scanned for any node
    whose opset domain is outside {"", "ai.onnx", "ai.onnx.ml"} and rejected
    (UNSUPPORTED_OP) — this package never registers a custom-op library, so
    a non-standard-domain op would otherwise either fail confusingly deep
    inside onnxruntime or (on a runtime that happened to have one
    registered) execute code this package never vetted. Reject, don't hope.
  - Tensor element counts are bounded (MAX_TENSOR_ELEMENTS) before any numpy
    allocation, as defense in depth alongside the overall transport-size cap.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import onnx
import onnx.checker
from onnx import TensorProto
import onnxruntime as ort

from gen.messages_pb2 import (
    Error,
    GraphNode,
    InitializerSpec,
    OpsetImport,
    Tensor,
    TensorSpec,
)

ort.disable_telemetry_events()

# The Axiom node transport caps a single message around ~4 MiB; capping the
# model itself at 3 MiB leaves headroom for the rest of the request/response
# envelope (input tensors, or an inspection result) within that limit.
MAX_MODEL_BYTES = 3 * 1024 * 1024  # 3 MiB

# Defense in depth: bounds a single tensor's element count independently of
# the overall transport cap, so a pathological shape (e.g. many small dims
# multiplying to something huge) fails fast with a clear error instead of
# attempting a giant allocation.
MAX_TENSOR_ELEMENTS = 5_000_000

MAX_LISTED_INITIALIZERS = 500
DEFAULT_MAX_GRAPH_NODES = 200
HARD_MAX_GRAPH_NODES = 2000

# Running any op outside these domains would require a custom-op library
# this package never registers — see the module docstring.
ALLOWED_DOMAINS = frozenset({"", "ai.onnx", "ai.onnx.ml"})

_INT_BOUNDS = {
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-2147483648, 2147483647),
    "int64": (-(2**63), 2**63 - 1),
    "uint8": (0, 255),
    "uint16": (0, 65535),
    "uint32": (0, 4294967295),
    "uint64": (0, 2**64 - 1),
}

_NP_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
}

_FLOAT_DTYPES = frozenset({"float16", "float32", "float64"})
_INT_DTYPES = frozenset(_INT_BOUNDS)

_NP_TO_DTYPE_STR = {
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
}


class OnnxToolsError(Exception):
    """A structured (code, message) failure a node can catch and turn
    directly into its output message's `error` field."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def to_error(exc: OnnxToolsError) -> Error:
    return Error(code=exc.code, message=exc.message)


def internal_error() -> Error:
    return Error(code="INTERNAL", message="internal error")


# ---------------------------------------------------------------------------
# Model decoding
# ---------------------------------------------------------------------------


def check_model_size(model_data: bytes) -> None:
    if not model_data:
        raise OnnxToolsError("MALFORMED_MODEL", "model_data is empty")
    if len(model_data) > MAX_MODEL_BYTES:
        raise OnnxToolsError(
            "MODEL_TOO_LARGE",
            f"model is {len(model_data)} bytes, exceeds the {MAX_MODEL_BYTES}-byte "
            "(3 MiB) cap for this package",
        )


def parse_model(model_data: bytes) -> onnx.ModelProto:
    """bytes -> onnx.ModelProto. Enforces the size cap first, then raises
    OnnxToolsError(MALFORMED_MODEL) on any parse failure."""
    check_model_size(model_data)
    try:
        return onnx.load_from_string(model_data)
    except Exception as exc:  # onnx raises assorted exception types on bad bytes
        raise OnnxToolsError("MALFORMED_MODEL", f"failed to parse ONNX model: {exc}") from exc


def iter_all_nodes(graph: onnx.GraphProto):
    """Yield every NodeProto in `graph`, recursively descending into every
    subgraph reachable via a GRAPH- or GRAPHS-typed node attribute (If's
    then_branch/else_branch, Loop's body, Scan's body, and any future
    control-flow op with a nested graph). Security-critical: a node hidden
    inside a subgraph is just as executable as a top-level one, so any code
    that must see "every op this model could run" (the domain guard below,
    and the operator inventory) has to walk this, not model.graph.node
    alone — a flat scan silently misses nested ops entirely."""
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from iter_all_nodes(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    yield from iter_all_nodes(sub)


def check_standard_domains_only(model: onnx.ModelProto) -> None:
    for node in iter_all_nodes(model.graph):
        if node.domain not in ALLOWED_DOMAINS:
            raise OnnxToolsError(
                "UNSUPPORTED_OP",
                f'node "{node.name or node.op_type}" (op_type={node.op_type}) uses '
                f'non-standard opset domain "{node.domain}"; only "", "ai.onnx", and '
                '"ai.onnx.ml" are supported',
            )


def create_session(model_data: bytes) -> ort.InferenceSession:
    """Parse + domain-check + build a deterministic, CPU-only,
    single-threaded onnxruntime session. Raises OnnxToolsError."""
    model = parse_model(model_data)
    check_standard_domains_only(model)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        return ort.InferenceSession(
            model_data, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise OnnxToolsError("MALFORMED_MODEL", f"onnxruntime failed to load model: {exc}") from exc


# ---------------------------------------------------------------------------
# Structural inspection helpers
# ---------------------------------------------------------------------------


def elem_type_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except (ValueError, TypeError):
        return "UNDEFINED"


def _dim_to_str(dim) -> str:
    which = dim.WhichOneof("value")
    if which == "dim_value":
        return str(dim.dim_value)
    if which == "dim_param":
        return dim.dim_param
    return "?"


def value_info_to_spec(vi) -> TensorSpec:
    tt = vi.type.tensor_type
    dims = [_dim_to_str(d) for d in tt.shape.dim]
    return TensorSpec(name=vi.name, elem_type=elem_type_name(tt.elem_type), dims=dims)


def graph_input_specs(model: onnx.ModelProto) -> list[TensorSpec]:
    """The model's TRUE inputs — graph.input entries that are not also
    initializers (some exporters redundantly list initializers as inputs;
    those are not something a caller supplies)."""
    initializer_names = {init.name for init in model.graph.initializer}
    return [
        value_info_to_spec(vi) for vi in model.graph.input if vi.name not in initializer_names
    ]


def graph_output_specs(model: onnx.ModelProto) -> list[TensorSpec]:
    return [value_info_to_spec(vi) for vi in model.graph.output]


def default_opset_version(model: onnx.ModelProto) -> int:
    for op in model.opset_import:
        if op.domain == "":
            return op.version
    return 0


def all_opset_imports(model: onnx.ModelProto) -> list[OpsetImport]:
    return [OpsetImport(domain=op.domain, version=op.version) for op in model.opset_import]


def initializer_element_count(init) -> int:
    count = 1
    for d in init.dims:
        count *= d
    return count


def graph_node_to_message(n) -> GraphNode:
    return GraphNode(
        name=n.name,
        op_type=n.op_type,
        inputs=list(n.input),
        outputs=list(n.output),
        domain=n.domain,
    )


def initializer_to_spec(init) -> InitializerSpec:
    return InitializerSpec(
        name=init.name,
        elem_type=elem_type_name(init.data_type),
        shape=list(init.dims),
        element_count=initializer_element_count(init),
    )


# ---------------------------------------------------------------------------
# Tensor <-> numpy marshaling
# ---------------------------------------------------------------------------


def tensor_to_numpy(t: Tensor) -> np.ndarray:
    """Input Tensor message -> numpy array matching its declared dtype and
    shape. Raises OnnxToolsError(INVALID_INPUT | TENSOR_MISMATCH)."""
    name = t.name or "<unnamed>"
    dtype = t.dtype or "float32"
    shape = tuple(t.shape)

    n_expected = 1
    for d in shape:
        if d < 0:
            raise OnnxToolsError("INVALID_INPUT", f'tensor "{name}": shape dims must be non-negative')
        n_expected *= d
    if n_expected > MAX_TENSOR_ELEMENTS:
        raise OnnxToolsError(
            "INVALID_INPUT",
            f'tensor "{name}": {n_expected} elements exceeds the {MAX_TENSOR_ELEMENTS}-element cap',
        )

    if dtype in _FLOAT_DTYPES:
        raw = list(t.float_data)
        _check_len(name, "float_data", raw, n_expected)
        arr = np.array(raw, dtype=np.float64).astype(_NP_DTYPE[dtype])
    elif dtype in _INT_DTYPES:
        raw = list(t.int_data)
        _check_len(name, "int_data", raw, n_expected)
        lo, hi = _INT_BOUNDS[dtype]
        for v in raw:
            if v < lo or v > hi:
                raise OnnxToolsError(
                    "INVALID_INPUT",
                    f'tensor "{name}": value {v} out of range for dtype "{dtype}" [{lo}, {hi}]',
                )
        arr = np.array(raw, dtype=_NP_DTYPE[dtype])
    elif dtype == "bool":
        raw = list(t.bool_data)
        _check_len(name, "bool_data", raw, n_expected)
        arr = np.array(raw, dtype=np.bool_)
    elif dtype == "string":
        raw = list(t.string_data)
        _check_len(name, "string_data", raw, n_expected)
        arr = np.array(raw, dtype=object)
    else:
        raise OnnxToolsError("INVALID_INPUT", f'tensor "{name}": unsupported dtype "{dtype}"')

    return arr.reshape(shape)


def _check_len(name: str, field: str, raw: list, n_expected: int) -> None:
    if len(raw) != n_expected:
        raise OnnxToolsError(
            "TENSOR_MISMATCH",
            f'tensor "{name}": {field} has {len(raw)} element(s), but shape requires {n_expected}',
        )


def numpy_to_tensor(name: str, arr: np.ndarray) -> Tensor:
    """numpy array (an onnxruntime output) -> output Tensor message."""
    arr = np.asarray(arr)
    shape = list(arr.shape)
    kind = arr.dtype.kind
    if kind == "f":
        dtype_str = _NP_TO_DTYPE_STR.get(arr.dtype, "float32")
        flat = [float(x) for x in arr.reshape(-1).tolist()]
        return Tensor(name=name, dtype=dtype_str, shape=shape, float_data=flat)
    if kind in ("i", "u"):
        dtype_str = _NP_TO_DTYPE_STR.get(arr.dtype, "int64")
        flat = [int(x) for x in arr.reshape(-1).tolist()]
        return Tensor(name=name, dtype=dtype_str, shape=shape, int_data=flat)
    if kind == "b":
        flat = [bool(x) for x in arr.reshape(-1).tolist()]
        return Tensor(name=name, dtype="bool", shape=shape, bool_data=flat)
    if kind in ("U", "O", "S"):
        flat = [x.decode() if isinstance(x, bytes) else str(x) for x in arr.reshape(-1).tolist()]
        return Tensor(name=name, dtype="string", shape=shape, string_data=flat)
    raise OnnxToolsError("INTERNAL", f'tensor "{name}": unsupported output dtype kind "{kind}"')


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_session(model_data: bytes, input_tensors: Iterable[Tensor]) -> tuple[list[str], list[np.ndarray]]:
    """Create a session, feed the supplied tensors, run every graph output.
    Returns (output_names, output_arrays) in the graph's declared output
    order. Raises OnnxToolsError."""
    session = create_session(model_data)
    feed = {}
    for t in input_tensors:
        if t.name in feed:
            raise OnnxToolsError("INVALID_INPUT", f'duplicate input tensor name "{t.name}"')
        feed[t.name] = tensor_to_numpy(t)
    output_names = [o.name for o in session.get_outputs()]
    try:
        results = session.run(output_names, feed)
    except OnnxToolsError:
        raise
    except Exception as exc:
        raise OnnxToolsError("TENSOR_MISMATCH", f"inference failed: {exc}") from exc
    return output_names, results
