import onnx
import onnx.checker

from gen.messages_pb2 import OnnxModel, ValidationResult
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, check_model_size, internal_error, to_error


def validate_model(ax: AxiomContext, input: OnnxModel) -> ValidationResult:
    """Check whether a base64 blob is a well-formed ONNX model, via
    onnx.checker after a protobuf parse. valid=false with one message per
    distinct problem in errors is this node doing its job correctly on a
    bad model — error is reserved for this node's own processing failures
    (e.g. MODEL_TOO_LARGE), never for "the model was invalid."
    """
    try:
        check_model_size(input.model_data)
    except OnnxToolsError as exc:
        ax.log.info("validate_model rejected input", code=exc.code)
        return ValidationResult(error=to_error(exc))

    try:
        try:
            model = onnx.load_from_string(input.model_data)
        except Exception as exc:
            return ValidationResult(valid=False, errors=[f"failed to parse: {exc}"])

        errors = []
        try:
            onnx.checker.check_model(model)
        except Exception as exc:
            errors.append(str(exc))

        if errors:
            return ValidationResult(valid=False, errors=errors)
        return ValidationResult(valid=True)
    except Exception as exc:
        ax.log.error("validate_model faulted", error=str(exc))
        return ValidationResult(error=internal_error())
