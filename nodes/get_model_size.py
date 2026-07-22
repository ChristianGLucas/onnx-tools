from gen.messages_pb2 import OnnxModel, ModelSizeInfo
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, initializer_element_count, internal_error, parse_model, to_error


def get_model_size(ax: AxiomContext, input: OnnxModel) -> ModelSizeInfo:
    """Compute aggregate size statistics without enumerating every
    initializer: total parameter count (sum of every initializer's element
    count), the serialized model's byte size, and the initializer count.
    """
    try:
        model = parse_model(input.model_data)
        total_parameters = sum(initializer_element_count(init) for init in model.graph.initializer)
        return ModelSizeInfo(
            total_parameters=total_parameters,
            model_size_bytes=len(input.model_data),
            initializer_count=len(model.graph.initializer),
        )
    except OnnxToolsError as exc:
        ax.log.info("get_model_size rejected input", code=exc.code)
        return ModelSizeInfo(error=to_error(exc))
    except Exception as exc:
        ax.log.error("get_model_size faulted", error=str(exc))
        return ModelSizeInfo(error=internal_error())
