from gen.messages_pb2 import GraphStructureRequest, GraphStructure
from gen.axiom_context import AxiomContext
from nodes._shared import (
    DEFAULT_MAX_GRAPH_NODES,
    HARD_MAX_GRAPH_NODES,
    OnnxToolsError,
    graph_node_to_message,
    internal_error,
    parse_model,
    to_error,
)


def get_graph_structure(ax: AxiomContext, input: GraphStructureRequest) -> GraphStructure:
    """Extract the computation graph as a DAG: each node's name, operator
    type, and the named tensor edges flowing in and out of it, plus the
    graph's own overall input/output tensor names. Capped at max_nodes
    entries (0 -> default 200; hard-capped at 2000 regardless of the
    caller's value).
    """
    try:
        if input.max_nodes < 0:
            raise OnnxToolsError("INVALID_ARGUMENT", "max_nodes must be non-negative")
        max_nodes = input.max_nodes if input.max_nodes > 0 else DEFAULT_MAX_GRAPH_NODES
        max_nodes = min(max_nodes, HARD_MAX_GRAPH_NODES)

        model = parse_model(input.model.model_data)
        total = len(model.graph.node)
        nodes = [graph_node_to_message(n) for n in list(model.graph.node)[:max_nodes]]
        return GraphStructure(
            nodes=nodes,
            graph_inputs=[vi.name for vi in model.graph.input],
            graph_outputs=[vi.name for vi in model.graph.output],
            truncated=total > len(nodes),
            total_node_count=total,
        )
    except OnnxToolsError as exc:
        ax.log.info("get_graph_structure rejected input", code=exc.code)
        return GraphStructure(error=to_error(exc))
    except Exception as exc:
        ax.log.error("get_graph_structure faulted", error=str(exc))
        return GraphStructure(error=internal_error())
