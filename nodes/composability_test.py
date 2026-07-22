"""Local composability checks: chain this package's own nodes exactly as a
compiled flow would wire them, in-process — the cheapest possible proof that
the envelope (OnnxModel) and result shapes actually compose before any
deploy-time flow-preview.
"""

from gen.messages_pb2 import (
    InferenceRequest,
    OnnxModel,
    OpsetCompatRequest,
    Tensor,
    TopKInferenceRequest,
)
from nodes.fixtures import build_tiny_mlp_model
from nodes.get_schema_contract import get_schema_contract
from nodes.check_opset_compatibility import check_opset_compatibility
from nodes.run_inference import run_inference
from nodes.run_inference_top_k import run_inference_top_k
from nodes.testkit import assert_ok, ax
from nodes.validate_model import validate_model


def test_validate_then_run_inference_chain():
    """ValidateModel -> (gate) -> RunInference: the natural "check before you
    call" flow edge."""
    model = OnnxModel(model_data=build_tiny_mlp_model())
    validation = validate_model(ax(), model)
    assert validation.valid is True

    req = InferenceRequest(
        model=model, inputs=[Tensor(name="X", dtype="float32", shape=[1, 3], float_data=[1, 1, 1])]
    )
    result = run_inference(ax(), req)
    assert_ok(result)
    assert list(result.outputs[0].float_data) == [0.0, 16.0]


def test_schema_contract_output_shape_matches_run_inference_output():
    """GetSchemaContract's declared output name matches what RunInference
    actually returns — proves the "agent learns the contract, then calls
    the model" flow edge is honest."""
    model = OnnxModel(model_data=build_tiny_mlp_model())
    schema = get_schema_contract(ax(), model)
    assert_ok(schema)
    assert len(schema.outputs) == 1
    declared_output_name = schema.outputs[0].name

    req = InferenceRequest(
        model=model, inputs=[Tensor(name="X", dtype="float32", shape=[1, 3], float_data=[1, 1, 1])]
    )
    result = run_inference(ax(), req)
    assert_ok(result)
    assert result.outputs[0].name == declared_output_name


def test_check_opset_compatibility_then_run_inference():
    """CheckOpsetCompatibility -> (gate) -> RunInference."""
    model = OnnxModel(model_data=build_tiny_mlp_model())
    compat = check_opset_compatibility(
        ax(), OpsetCompatRequest(model=model, target_opset=18)
    )
    assert_ok(compat)
    assert compat.compatible is True

    req = InferenceRequest(
        model=model, inputs=[Tensor(name="X", dtype="float32", shape=[1, 3], float_data=[2, 0, 1])]
    )
    result = run_inference(ax(), req)
    assert_ok(result)
    assert list(result.outputs[0].float_data) == [0.0, 15.0]


def test_run_inference_and_top_k_agree_on_same_model_and_input():
    """RunInference's raw output and RunInferenceTopK's reduction of it must
    describe the SAME underlying computation — cross-checked directly."""
    model = OnnxModel(model_data=build_tiny_mlp_model())
    inputs = [Tensor(name="X", dtype="float32", shape=[1, 3], float_data=[1, 1, 1])]

    full = run_inference(ax(), InferenceRequest(model=model, inputs=inputs))
    assert_ok(full)
    raw_scores = list(full.outputs[0].float_data)

    topk = run_inference_top_k(ax(), TopKInferenceRequest(model=model, inputs=inputs, k=2))
    assert_ok(topk)
    # topk's scores, re-sorted, must be exactly the same multiset as raw_scores.
    assert sorted(topk.scores) == sorted(raw_scores)
