from gen.messages_pb2 import OnnxModel, Tensor, TopKInferenceRequest
from nodes.fixtures import build_tiny_mlp_model
from nodes.run_inference_top_k import run_inference_top_k
from nodes.testkit import assert_error, assert_ok, ax


def _mlp_req(**kwargs) -> TopKInferenceRequest:
    model = build_tiny_mlp_model()
    return TopKInferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[Tensor(name="X", dtype="float32", shape=[1, 3], float_data=[1, 1, 1])],
        **kwargs,
    )


def test_topk_2_of_2_sorted_descending():
    # Hand-computed MLP output for X=[1,1,1] is [0, 16] (see fixtures.py) —
    # top-2 must therefore be index 1 (score 16) then index 0 (score 0).
    result = run_inference_top_k(ax(), _mlp_req(k=2))
    assert_ok(result)
    assert result.output_name == "Y"
    assert list(result.indices) == [1, 0]
    assert list(result.scores) == [16.0, 0.0]


def test_topk_default_k_is_one():
    result = run_inference_top_k(ax(), _mlp_req(k=0))
    assert_ok(result)
    assert list(result.indices) == [1]
    assert list(result.scores) == [16.0]


def test_topk_k_larger_than_output_size_is_clamped():
    result = run_inference_top_k(ax(), _mlp_req(k=50))
    assert_ok(result)
    assert len(result.indices) == 2  # the output only has 2 elements


def test_explicit_output_name_selects_same_output():
    result = run_inference_top_k(ax(), _mlp_req(k=1, output_name="Y"))
    assert_ok(result)
    assert result.output_name == "Y"
    assert list(result.indices) == [1]


def test_unknown_output_name_is_invalid_argument():
    result = run_inference_top_k(ax(), _mlp_req(k=1, output_name="NoSuchOutput"))
    assert_error(result, "INVALID_ARGUMENT")


def test_negative_k_is_invalid_argument():
    result = run_inference_top_k(ax(), _mlp_req(k=-1))
    assert_error(result, "INVALID_ARGUMENT")
