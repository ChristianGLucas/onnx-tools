from gen.messages_pb2 import OnnxModel, OperatorInventory, OperatorCount
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, internal_error, parse_model, to_error


def list_operators(ax: AxiomContext, input: OnnxModel) -> OperatorInventory:
    """Inventory which operator types a model's graph uses and how many
    graph nodes use each one, plus the total node count.
    """
    try:
        model = parse_model(input.model_data)
        counts: dict[str, int] = {}
        for node in model.graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        operators = [
            OperatorCount(op_type=op_type, count=count)
            for op_type, count in sorted(counts.items())
        ]
        return OperatorInventory(operators=operators, total_nodes=len(model.graph.node))
    except OnnxToolsError as exc:
        ax.log.info("list_operators rejected input", code=exc.code)
        return OperatorInventory(error=to_error(exc))
    except Exception as exc:
        ax.log.error("list_operators faulted", error=str(exc))
        return OperatorInventory(error=internal_error())
