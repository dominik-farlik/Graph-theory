import networkx as nx
from matplotlib import pyplot as plt

from edge.edge import Direction


class DrawMixin:


    def draw(self):
        G = nx.DiGraph() if self.orientovany else nx.Graph()
        for node in self.nodes.values():
            G.add_node(node.name)

        for edge in self.edges:
            if edge.direction == Direction.TO:
                u, v = edge.node1.name, edge.node2.name
            else:
                u, v = edge.node2.name, edge.node1.name
            G.add_edge(u, v, label=edge.name or "", weight=edge.length or 1)

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos,
                with_labels=True,
                node_color="lightblue",
                node_size=1500,
                arrows=self.orientovany,
                font_weight="bold")

        edge_labels = {
            (u, v): f"{d['label']} ({d['weight']})" if d['label'] else d['weight']
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        plt.show()