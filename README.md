# onnx-tools

Deterministic ONNX model inspection and inference for the
[Axiom](https://axiomide.com) marketplace, published as
`christiangeorgelucas/onnx-tools`.

Run a caller-supplied ONNX model (base64) on caller-supplied input tensors
and get back real output tensors, or inspect its structure — inputs/outputs
schema, metadata, opset, operator inventory, initializer/parameter counts,
the computation graph, and well-formedness — without an agent shipping a
whole ML stack. Wraps [onnx](https://github.com/onnx/onnx) (Apache-2.0,
structural parsing/validation via `onnx.checker`) and
[onnxruntime](https://github.com/microsoft/onnxruntime) (MIT, the reference
ONNX inference engine).

The **model** and every **input tensor** are direct caller inputs — nothing
is ever fetched by URL or downloaded, and no node makes a network call.

## Determinism & safety

- **CPU-only**: every inference session is created with
  `providers=["CPUExecutionProvider"]` explicitly — never onnxruntime's own
  provider auto-selection.
- **Single-threaded, sequential**: `intra_op_num_threads` /
  `inter_op_num_threads` = 1, `ExecutionMode.ORT_SEQUENTIAL` — a fixed model
  + fixed input produces byte-identical output on every call (parallel
  reduction order is a real source of nondeterminism onnxruntime otherwise
  permits).
- **No custom ops, ever**: before a session is created, the graph is
  scanned for any node whose opset domain is outside `""`, `"ai.onnx"`, or
  `"ai.onnx.ml"` and rejected as `UNSUPPORTED_OP`. This package never
  registers a custom-op library, so a non-standard-domain op is refused
  outright rather than executed.
- **Telemetry disabled** at import (`onnxruntime.disable_telemetry_events()`).
- Malformed/unparseable input never crashes a node — every output message
  carries an `error` field, unset on success. Size/memory/DoS limits are the
  Axiom platform's job, not this package's.

## Nodes

| Node | Does |
|---|---|
| `RunInference` | Run the model on named input tensors → named output tensors |
| `RunInferenceTopK` | RunInference, then reduce one output to its top-k indices/scores |
| `ListModelInputs` | Name/type/dims of the model's true graph inputs (dynamic axes included) |
| `ListModelOutputs` | Name/type/dims of the model's graph outputs |
| `GetModelMetadata` | producer/version/ir_version/domain/doc_string/graph_name/opset_imports |
| `ListOperators` | Operator-type inventory + counts — "what does this model do" |
| `ListInitializers` | Weight-tensor inventory: name/type/shape/count (never raw values) |
| `GetModelSize` | Aggregate total parameter count + serialized byte size |
| `ValidateModel` | Well-formed-ONNX check via `onnx.checker`, with per-problem messages |
| `GetGraphStructure` | The computation graph as a DAG (node name/op/edges) |
| `GetOpsetVersion` | The model's default-domain opset version + every opset import |
| `GetSchemaContract` | Inputs + outputs + opset in one call — the "how do I call this" contract |
| `CheckOpsetCompatibility` | Is the model's opset ≤ a target opset (backward-compatibility check) |

See `messages/messages.proto` for the exact field-level contract of every
request/response message, and `axiom.yaml` for each node's full description.

## The envelope

Every node consumes `OnnxModel{ bytes model_data }` — either directly, or
embedded in a small request message alongside tensors/options
(`InferenceRequest`, `TopKInferenceRequest`, `GraphStructureRequest`,
`OpsetCompatRequest`). `Tensor` is the single shared shape for both request
and response tensors: `name`, `dtype` (selects which typed data field is
populated), `shape`, and one of `float_data` / `int_data` / `bool_data` /
`string_data`.

## Error contract

`Error{ code, message }`, unset on success:

`MALFORMED_MODEL` · `UNSUPPORTED_OP` · `TENSOR_MISMATCH`
· `INVALID_INPUT` · `INVALID_ARGUMENT` · `INTERNAL`

`ValidateModel` is the one node where `valid=false` + `errors` is the
**correct, intended result** on a bad model, not a failure — `error` there
is reserved for the node's own processing failures (e.g. empty `model_data`).

## Tests

```bash
axiom test
```

A golden test per node against hand-built ONNX fixtures (`nodes/fixtures.py`
— built programmatically with `onnx.helper`, not committed binaries, so
every ground-truth value is visible in the diff), an independent-oracle
suite (`oracle_test.py`) that checks `RunInference` against a from-scratch
pure-Python re-implementation of the fixture's Gemm+Relu math (zero
dependency on onnx/onnxruntime/numpy), a security suite (`security_test.py`)
covering the non-standard-domain guard and malformed/adversarial input
across every node, and a composability suite (`composability_test.py`)
chaining this package's own nodes exactly as a compiled flow would wire
them.

## Licence

MIT — see [LICENSE](LICENSE).

Wraps onnx (Apache-2.0) and onnxruntime (MIT); full transitive closure
(numpy, protobuf, flatbuffers, packaging, typing_extensions, ml_dtypes) is
permissive — see [requirements.txt](requirements.txt) for the full audit.
No copyleft anywhere in the tree.

Built for the Axiom marketplace.
