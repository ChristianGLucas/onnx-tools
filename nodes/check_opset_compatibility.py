from gen.messages_pb2 import OpsetCompatRequest, OpsetCompatResult
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, default_opset_version, internal_error, parse_model, to_error


def check_opset_compatibility(ax: AxiomContext, input: OpsetCompatRequest) -> OpsetCompatResult:
    """Check whether a model is compatible with a given target opset
    version: compatible when the model's own default-domain opset version
    is less than or equal to target_opset (ONNX opsets are
    backward-compatible within a domain).
    """
    try:
        if input.target_opset <= 0:
            raise OnnxToolsError("INVALID_ARGUMENT", "target_opset must be positive")

        model = parse_model(input.model.model_data)
        model_opset = default_opset_version(model)
        target = input.target_opset
        compatible = model_opset <= target
        note = (
            f"model opset {model_opset} <= target {target}: compatible"
            if compatible
            else (
                f"model opset {model_opset} > target {target}: the model was authored "
                "against a newer opset than the target supports"
            )
        )
        return OpsetCompatResult(
            compatible=compatible, model_opset=model_opset, target_opset=target, note=note
        )
    except OnnxToolsError as exc:
        ax.log.info("check_opset_compatibility rejected input", code=exc.code)
        return OpsetCompatResult(error=to_error(exc))
    except Exception as exc:
        ax.log.error("check_opset_compatibility faulted", error=str(exc))
        return OpsetCompatResult(error=internal_error())
