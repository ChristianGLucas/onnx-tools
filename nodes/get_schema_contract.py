from gen.messages_pb2 import OnnxModel, SchemaContract
from gen.axiom_context import AxiomContext
from nodes._shared import (
    OnnxToolsError,
    default_opset_version,
    graph_input_specs,
    graph_output_specs,
    internal_error,
    parse_model,
    to_error,
)


def get_schema_contract(ax: AxiomContext, input: OnnxModel) -> SchemaContract:
    """Get the model's full input/output shape contract in one call — the
    union of ListModelInputs + ListModelOutputs + the default-domain opset
    version and graph name — so a calling agent knows exactly how to build
    a RunInference request without three separate calls.
    """
    try:
        model = parse_model(input.model_data)
        return SchemaContract(
            inputs=graph_input_specs(model),
            outputs=graph_output_specs(model),
            opset_version=default_opset_version(model),
            graph_name=model.graph.name,
        )
    except OnnxToolsError as exc:
        ax.log.info("get_schema_contract rejected input", code=exc.code)
        return SchemaContract(error=to_error(exc))
    except Exception as exc:
        ax.log.error("get_schema_contract faulted", error=str(exc))
        return SchemaContract(error=internal_error())
