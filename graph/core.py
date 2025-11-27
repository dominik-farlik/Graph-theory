from typing import Optional, Any

from graph.degrees import DegreesMixin
from graph.draw import DrawMixin
from graph.export import ExportMixin
from graph.matrices import MatrixMixin
from graph.properties import PropertiesMixin
from graph.sets import SetsMixin
from graph.traversal import TraversalMixin
from node.node import Node
from edge.edge import Edge, Direction

TYPE = 0
NODE_NAME = 1
NODE_1 = 1
DIRECTION = 2
NODE_2 = 3


class Graph(MatrixMixin, PropertiesMixin, DegreesMixin, SetsMixin, TraversalMixin, ExportMixin, DrawMixin):
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.edge_name = None
        self.edge_length = None
        self.edge_structure = False

    def __repr__(self):
        return "\n".join(repr(edge) for edge in self.edges)

    def add_node(self, line: list) -> None:
        node = Node(line[NODE_NAME])
        self.nodes[node.name] = node

    def add_edge(self, line: list) -> None:
        node1 = self.nodes[line[NODE_1]]
        node2 = self.nodes[line[NODE_2]]
        direction = Direction(line[DIRECTION])
        name, length = self.parse_edge_name_length(line)

        edge = Edge(node1, node2, direction, name, length)
        self.edges.append(edge)

    def init_edge_structure(self, line) -> None:
        if len(line) == 5:
            if line[4].startswith(':'):
                self.edge_name = 4
            else:
                self.edge_length = 4
        elif len(line) == 6:
            self.edge_length = 4
            self.edge_name = 5

        self.edge_structure = True

    def parse_edge_name_length(self, line) -> tuple[str, Optional[int]]:
        name = ""
        length = None

        if not self.edge_structure:
            self.init_edge_structure(line)

        if self.edge_name:
            name = line[self.edge_name][1:]
        if self.edge_length:
            length = line[self.edge_length]

        return name, length

def load_graph_from_file(filename: str) -> Graph:
    g = Graph()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue

            line = line.split(" ")
            if line[TYPE] == "u":
                g.add_node(line)

            elif line[TYPE] == "h":
                g.add_edge(line)
    return g
