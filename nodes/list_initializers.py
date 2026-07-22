from gen.messages_pb2 import OnnxModel, InitializerInventory
from gen.axiom_context import AxiomContext
from nodes._shared import (
    MAX_LISTED_INITIALIZERS,
    OnnxToolsError,
    initializer_element_count,
    initializer_to_spec,
    internal_error,
    parse_model,
    to_error,
)


def list_initializers(ax: AxiomContext, input: OnnxModel) -> InitializerInventory:
    """Inventory the model's initializers (baked-in learned weight
    tensors): name, ONNX element type, shape, and element count — never the
    raw values. Listed entries are capped at 500; total_parameters and
    total_initializer_count always cover the model's FULL initializer set.
    """
    try:
        model = parse_model(input.model_data)
        inits = list(model.graph.initializer)
        total_parameters = sum(initializer_element_count(init) for init in inits)
        listed = inits[:MAX_LISTED_INITIALIZERS]
        specs = [initializer_to_spec(init) for init in listed]
        return InitializerInventory(
            initializers=specs,
            total_parameters=total_parameters,
            total_initializer_count=len(inits),
            truncated=len(inits) > MAX_LISTED_INITIALIZERS,
        )
    except OnnxToolsError as exc:
        ax.log.info("list_initializers rejected input", code=exc.code)
        return InitializerInventory(error=to_error(exc))
    except Exception as exc:
        ax.log.error("list_initializers faulted", error=str(exc))
        return InitializerInventory(error=internal_error())
