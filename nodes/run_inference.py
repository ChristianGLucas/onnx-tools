from gen.messages_pb2 import InferenceRequest, InferenceResponse
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, internal_error, numpy_to_tensor, run_session, to_error


def run_inference(ax: AxiomContext, input: InferenceRequest) -> InferenceResponse:
    """Run a caller-supplied ONNX model on caller-supplied named input
    tensors and return every named output tensor the graph produces, in the
    graph's declared output order. CPU-only, single-threaded, deterministic.
    See RunInference's axiom.yaml description for the full error contract.
    """
    try:
        names, results = run_session(input.model.model_data, input.inputs)
        outputs = [numpy_to_tensor(n, r) for n, r in zip(names, results)]
        return InferenceResponse(outputs=outputs)
    except OnnxToolsError as exc:
        ax.log.info("run_inference rejected input", code=exc.code)
        return InferenceResponse(error=to_error(exc))
    except Exception as exc:
        ax.log.error("run_inference faulted", error=str(exc))
        return InferenceResponse(error=internal_error())
