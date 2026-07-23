from gen.messages_pb2 import GraphStructureRequest, GraphStructure
from gen.axiom_context import AxiomContext
from nodes._shared import (
    OnnxToolsError,
    graph_node_to_message,
    internal_error,
    parse_model,
    to_error,
)


def get_graph_structure(ax: AxiomContext, input: GraphStructureRequest) -> GraphStructure:
    """Extract the computation graph as a DAG: each node's name, operator
    type, and the named tensor edges flowing in and out of it, plus the
    graph's own overall input/output tensor names.
    """
    try:
        model = parse_model(input.model.model_data)
        nodes = [graph_node_to_message(n) for n in model.graph.node]
        return GraphStructure(
            nodes=nodes,
            graph_inputs=[vi.name for vi in model.graph.input],
            graph_outputs=[vi.name for vi in model.graph.output],
        )
    except OnnxToolsError as exc:
        ax.log.info("get_graph_structure rejected input", code=exc.code)
        return GraphStructure(error=to_error(exc))
    except Exception as exc:
        ax.log.error("get_graph_structure faulted", error=str(exc))
        return GraphStructure(error=internal_error())
