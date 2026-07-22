import numpy as np

from gen.messages_pb2 import TopKInferenceRequest, TopKResult
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, internal_error, run_session, to_error


def run_inference_top_k(ax: AxiomContext, input: TopKInferenceRequest) -> TopKResult:
    """Run inference like RunInference, then reduce one selected output
    tensor to its top-k indices and scores (descending), for the common
    classifier-head convenience case. See axiom.yaml for the full contract.
    """
    try:
        if input.k < 0:
            raise OnnxToolsError("INVALID_ARGUMENT", "k must be non-negative")
        k = input.k if input.k > 0 else 1

        names, results = run_session(input.model.model_data, input.inputs)
        if input.output_name:
            if input.output_name not in names:
                raise OnnxToolsError(
                    "INVALID_ARGUMENT",
                    f'no such output "{input.output_name}"; graph outputs are {names}',
                )
            idx = names.index(input.output_name)
        else:
            if not names:
                raise OnnxToolsError("INTERNAL", "model graph declares no outputs")
            idx = 0
        chosen_name = names[idx]

        arr = np.asarray(results[idx]).astype(np.float64).reshape(-1)
        if arr.size == 0:
            raise OnnxToolsError("INVALID_ARGUMENT", f'output "{chosen_name}" is empty; cannot compute top-k')

        k_eff = min(k, arr.size)
        # Stable sort so ties keep ascending-index order — deterministic.
        order = np.argsort(-arr, kind="stable")[:k_eff]
        indices = [int(i) for i in order]
        scores = [float(arr[i]) for i in order]
        return TopKResult(output_name=chosen_name, indices=indices, scores=scores)
    except OnnxToolsError as exc:
        ax.log.info("run_inference_top_k rejected input", code=exc.code)
        return TopKResult(error=to_error(exc))
    except Exception as exc:
        ax.log.error("run_inference_top_k faulted", error=str(exc))
        return TopKResult(error=internal_error())
