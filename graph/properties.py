import networkx as nx

from edge.edge import Direction, Edge
from node.node import Node


class PropertiesMixin:
    edges: list[Edge]
    nodes: dict[str, Node]
    orientovany: bool

    @staticmethod
    def _reach(start: Node, neigh_fn) -> set[Node]:
        seen = set()
        stack = [start]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            stack.extend(neigh_fn(v) - seen)
        return seen

    @property
    def orientovany(self):
        if self.edges[0].direction == Direction.NONE:
            return False
        return True

    @property
    def hranove_ohodnoceny(self):
        if self.edges[0].length:
            return True
        return False

    @property
    def prosty(self):
        if not self.orientovany:
            if len(self.edges) == len(set(self.edges)):
                return True
        return False

    @property
    def multigraf(self):
        return not self.prosty

    @property
    def jednoduchy(self):
        if self.prosty:
            return not any(edge.node1 == edge.node2 for edge in self.edges)
        return False

    @property
    def souvisly(self):
        if not self.orientovany:
            start = next(iter(self.nodes.values()))
            seen = self._reach(start, self.get_mnozina_sousedu)
            return len(seen) == len(self.nodes)
        return False

    @property
    def slabe_souvisly(self):
        if self.orientovany:
            start = next(iter(self.nodes.values()))
            seen = self._reach(start, self.get_mnozina_sousedu)
            return len(seen) == len(self.nodes)
        return False

    @property
    def silne_souvisly(self) -> bool:
        if self.orientovany:
            start = next(iter(self.nodes.values()))
            seen_fwd = self._reach(start, self.get_mnozina_nasledniku)

            if len(seen_fwd) != len(self.nodes):
                return False

            seen_rev = self._reach(start, self.get_mnozina_predchudcu)
            return len(seen_rev) == len(self.nodes)

        return False

    @property
    def rovinny(self) -> bool:
        G = nx.Graph()

        for node in self.nodes.values():
            G.add_node(node.name)

        for edge in self.edges:
            G.add_edge(edge.node1.name, edge.node2.name)

        is_planar, _ = nx.check_planarity(G)
        return is_planar

    @property
    def uplny(self):
        if not self.orientovany:
            nodes = list(self.nodes.values())

            for i in range(len(self.nodes)):
                for j in range(i + 1, len(self.nodes)):
                    u, v = nodes[i], nodes[j]
                    if not any(
                            (e.node1 == u and e.node2 == v) or (e.node1 == v and e.node2 == u)
                            for e in self.edges
                    ):
                        return False
            return True
        return False

    @property
    def regularni(self):
        stupne = [self.get_stupen(u) for u in self.nodes.values()]
        return len(set(stupne)) == 1

    @property
    def bipartitni(self) -> bool:
        neighbours = {u: set() for u in self.nodes.values()}
        for e in self.edges:
            neighbours[e.node1].add(e.node2)
            neighbours[e.node2].add(e.node1)

        barva = {}

        for start in self.nodes.values():
            if start in barva:
                continue
            barva[start] = 0
            stack = [start]

            while stack:
                u = stack.pop()
                for v in neighbours[u]:
                    if v not in barva:
                        barva[v] = 1 - barva[u]
                        stack.append(v)
                    elif barva[v] == barva[u]:
                        return False

        return True

    def print_properties(self) -> None:
        print(f"Orientovaný: {self.orientovany}\n"
              f"Hranově ohodnocený: {self.hranove_ohodnoceny}\n"
              f"Prostý: {self.prosty}\nMultigraf: {self.multigraf}\n"
              f"Jednoduchý: {self.jednoduchy}\n"
              f"Souvislý: {self.souvisly}\n"
              f"Slabě souvislý: {self.slabe_souvisly}\n"
              f"Silně souvislý: {self.silne_souvisly}\n"
              f"Rovinný: {self.rovinny}\n"
              f"Úplný: {self.uplny}\n"
              f"Regulární: {self.regularni}\n"
              f"Bipartitní: {self.bipartitni}\n")