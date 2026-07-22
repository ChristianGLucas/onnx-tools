from gen.messages_pb2 import InferenceRequest, OnnxModel, Tensor
from nodes.fixtures import build_add_model, build_custom_domain_model, build_tiny_mlp_model
from nodes.run_inference import run_inference
from nodes.testkit import assert_error, assert_ok, ax


def test_add_model_matches_hand_computed_sum():
    model = build_add_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="A", dtype="float32", shape=[3], float_data=[1, 2, 3]),
            Tensor(name="B", dtype="float32", shape=[3], float_data=[4, 5, 6]),
        ],
    )
    result = run_inference(ax(), req)
    assert_ok(result)
    assert len(result.outputs) == 1
    out = result.outputs[0]
    assert out.name == "Y"
    assert list(out.shape) == [3]
    assert list(out.float_data) == [5.0, 7.0, 9.0]


def test_mlp_model_relu_clips_negative_class():
    model = build_tiny_mlp_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[Tensor(name="X", dtype="float32", shape=[1, 3], float_data=[1, 1, 1])],
    )
    result = run_inference(ax(), req)
    assert_ok(result)
    out = result.outputs[0]
    assert list(out.shape) == [1, 2]
    # Hand-computed in fixtures.py: gemm=[-4,16] -> relu=[0,16].
    assert list(out.float_data) == [0.0, 16.0]


def test_mlp_second_input_matches_hand_computed():
    model = build_tiny_mlp_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[Tensor(name="X", dtype="float32", shape=[1, 3], float_data=[2, 0, 1])],
    )
    result = run_inference(ax(), req)
    assert_ok(result)
    assert list(result.outputs[0].float_data) == [0.0, 15.0]


def test_repeat_invocation_is_deterministic():
    model = build_add_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="A", dtype="float32", shape=[3], float_data=[1, 2, 3]),
            Tensor(name="B", dtype="float32", shape=[3], float_data=[4, 5, 6]),
        ],
    )
    r1 = run_inference(ax(), req)
    r2 = run_inference(ax(), req)
    assert list(r1.outputs[0].float_data) == list(r2.outputs[0].float_data)


def test_missing_input_tensor_is_tensor_mismatch():
    model = build_add_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[Tensor(name="A", dtype="float32", shape=[3], float_data=[1, 2, 3])],
        # "B" deliberately omitted
    )
    result = run_inference(ax(), req)
    assert_error(result, "TENSOR_MISMATCH")


def test_shape_data_length_mismatch_is_tensor_mismatch():
    model = build_add_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="A", dtype="float32", shape=[3], float_data=[1, 2]),  # 2 values, shape wants 3
            Tensor(name="B", dtype="float32", shape=[3], float_data=[4, 5, 6]),
        ],
    )
    result = run_inference(ax(), req)
    assert_error(result, "TENSOR_MISMATCH")


def test_duplicate_input_tensor_name_is_invalid_input():
    model = build_add_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="A", dtype="float32", shape=[3], float_data=[1, 2, 3]),
            Tensor(name="A", dtype="float32", shape=[3], float_data=[7, 8, 9]),
        ],
    )
    result = run_inference(ax(), req)
    assert_error(result, "INVALID_INPUT")


def test_malformed_model_bytes():
    req = InferenceRequest(model=OnnxModel(model_data=b"not an onnx model"), inputs=[])
    result = run_inference(ax(), req)
    assert_error(result, "MALFORMED_MODEL")


def test_empty_model_bytes():
    req = InferenceRequest(model=OnnxModel(model_data=b""), inputs=[])
    result = run_inference(ax(), req)
    assert_error(result, "MALFORMED_MODEL")


def test_oversized_model_is_model_too_large():
    huge = b"\x00" * (3 * 1024 * 1024 + 1)
    req = InferenceRequest(model=OnnxModel(model_data=huge), inputs=[])
    result = run_inference(ax(), req)
    assert_error(result, "MODEL_TOO_LARGE")


def test_custom_domain_op_is_unsupported_op():
    model = build_custom_domain_model()
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[Tensor(name="X", dtype="float32", shape=[1], float_data=[1.0])],
    )
    result = run_inference(ax(), req)
    assert_error(result, "UNSUPPORTED_OP")


def test_out_of_range_int_value_is_invalid_input():
    model = build_add_model()  # dtype doesn't need to match graph exactly to trigger our own pre-check
    req = InferenceRequest(
        model=OnnxModel(model_data=model),
        inputs=[
            Tensor(name="A", dtype="int8", shape=[3], int_data=[1, 2, 300]),  # 300 > int8 max
            Tensor(name="B", dtype="float32", shape=[3], float_data=[4, 5, 6]),
        ],
    )
    result = run_inference(ax(), req)
    assert_error(result, "INVALID_INPUT")
