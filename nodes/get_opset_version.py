from gen.messages_pb2 import OnnxModel, OpsetInfo
from gen.axiom_context import AxiomContext
from nodes._shared import (
    OnnxToolsError,
    all_opset_imports,
    default_opset_version,
    internal_error,
    parse_model,
    to_error,
)


def get_opset_version(ax: AxiomContext, input: OnnxModel) -> OpsetInfo:
    """Detect a model's opset version: the default ("ai.onnx") domain's
    opset_version, plus every opset import (including any non-default
    domains) the model declares.
    """
    try:
        model = parse_model(input.model_data)
        return OpsetInfo(
            opset_version=default_opset_version(model),
            all_opsets=all_opset_imports(model),
        )
    except OnnxToolsError as exc:
        ax.log.info("get_opset_version rejected input", code=exc.code)
        return OpsetInfo(error=to_error(exc))
    except Exception as exc:
        ax.log.error("get_opset_version faulted", error=str(exc))
        return OpsetInfo(error=internal_error())
