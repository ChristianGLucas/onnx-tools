from gen.messages_pb2 import OnnxModel, ModelMetadata
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, all_opset_imports, internal_error, parse_model, to_error


def get_model_metadata(ax: AxiomContext, input: OnnxModel) -> ModelMetadata:
    """Read a model's metadata fields: producer_name, producer_version,
    ir_version, domain, doc_string, graph_name, model_version, and every
    opset import the model declares.
    """
    try:
        model = parse_model(input.model_data)
        return ModelMetadata(
            producer_name=model.producer_name,
            producer_version=model.producer_version,
            ir_version=model.ir_version,
            domain=model.domain,
            doc_string=model.doc_string,
            graph_name=model.graph.name,
            model_version=model.model_version,
            opset_imports=all_opset_imports(model),
        )
    except OnnxToolsError as exc:
        ax.log.info("get_model_metadata rejected input", code=exc.code)
        return ModelMetadata(error=to_error(exc))
    except Exception as exc:
        ax.log.error("get_model_metadata faulted", error=str(exc))
        return ModelMetadata(error=internal_error())
