from gen.messages_pb2 import OnnxModel, OperatorInventory, OperatorCount
from gen.axiom_context import AxiomContext
from nodes._shared import OnnxToolsError, internal_error, iter_all_nodes, parse_model, to_error


def list_operators(ax: AxiomContext, input: OnnxModel) -> OperatorInventory:
    """Inventory which operator types a model's graph uses and how many
    graph nodes use each one, plus the total node count. Counts every node
    reachable in the model, including ones nested inside If/Loop/Scan
    subgraphs — not just the top-level graph.
    """
    try:
        model = parse_model(input.model_data)
        counts: dict[str, int] = {}
        total_nodes = 0
        for node in iter_all_nodes(model.graph):
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
            total_nodes += 1
        operators = [
            OperatorCount(op_type=op_type, count=count)
            for op_type, count in sorted(counts.items())
        ]
        return OperatorInventory(operators=operators, total_nodes=total_nodes)
    except OnnxToolsError as exc:
        ax.log.info("list_operators rejected input", code=exc.code)
        return OperatorInventory(error=to_error(exc))
    except Exception as exc:
        ax.log.error("list_operators faulted", error=str(exc))
        return OperatorInventory(error=internal_error())
