from gen.messages_pb2 import OnnxModel, TensorSpecList
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, graph_input_specs, internal_error, parse_model, to_error


def list_model_inputs(ax: AxiomContext, input: OnnxModel) -> TensorSpecList:
    """List a model's true graph inputs (excluding initializers some
    exporters redundantly list as inputs): each entry's name, ONNX element
    type, and dims, with dynamic axes reported as their symbolic name or
    "?" when unnamed and unfixed.
    """
    try:
        model = parse_model(input.model_data)
        return TensorSpecList(specs=graph_input_specs(model))
    except OnnxToolsError as exc:
        ax.log.info("list_model_inputs rejected input", code=exc.code)
        return TensorSpecList(error=to_error(exc))
    except Exception as exc:
        ax.log.error("list_model_inputs faulted", error=str(exc))
        return TensorSpecList(error=internal_error())
