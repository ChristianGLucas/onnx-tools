from gen.messages_pb2 import OnnxModel, TensorSpecList
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, graph_output_specs, internal_error, parse_model, to_error


def list_model_outputs(ax: AxiomContext, input: OnnxModel) -> TensorSpecList:
    """List a model's graph outputs: each entry's name, ONNX element type,
    and dims (dynamic axes as their symbolic name or "?"), in the graph's
    declared output order — the order RunInference returns tensors in.
    """
    try:
        model = parse_model(input.model_data)
        return TensorSpecList(specs=graph_output_specs(model))
    except OnnxToolsError as exc:
        ax.log.info("list_model_outputs rejected input", code=exc.code)
        return TensorSpecList(error=to_error(exc))
    except Exception as exc:
        ax.log.error("list_model_outputs faulted", error=str(exc))
        return TensorSpecList(error=internal_error())
