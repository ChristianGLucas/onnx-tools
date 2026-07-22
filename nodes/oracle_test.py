"""Independent-oracle tests: RunInference's output is checked against a
hand-written pure-Python re-implementation of Gemm+Relu (no numpy, no onnx,
no onnxruntime) — genuinely independent of the wrapped library, not a
round-trip through the same code path. See fixtures.py's MLP_W/MLP_B for
the exact weights this oracle uses.
"""

import random

from gen.messages_pb2 import InferenceRequest, OnnxModel, Tensor
from nodes.fixtures import MLP_B, MLP_W, build_tiny_mlp_model
from nodes.run_inference import run_inference
from nodes.testkit import assert_ok, ax


def _pure_python_mlp(x: list[float]) -> list[float]:
    """Y = Relu(x @ W^T + B), computed with plain Python loops — zero
    dependency on onnx/onnxruntime/numpy."""
    w = MLP_W.tolist()
    b = MLP_B.tolist()
    gemm = [sum(x[j] * w[i][j] for j in range(len(x))) + b[i] for i in range(len(w))]
    return [max(v, 0.0) for v in gemm]


def test_run_inference_matches_independent_pure_python_reimplementation():
    model = build_tiny_mlp_model()
    rng = random.Random(42)
    for _ in range(20):
        x = [rng.uniform(-5, 5) for _ in range(3)]
        expected = _pure_python_mlp(x)

        req = InferenceRequest(
            model=OnnxModel(model_data=model),
            inputs=[Tensor(name="X", dtype="float32", shape=[1, 3], float_data=x)],
        )
        result = run_inference(ax(), req)
        assert_ok(result)
        actual = list(result.outputs[0].float_data)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            # float32 round-trip tolerance
            assert abs(a - e) < 1e-3, f"x={x}: expected {expected}, got {actual}"
