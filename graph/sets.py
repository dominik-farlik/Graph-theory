from typing import Any

from edge.edge import Direction, Edge
from node.node import Node


class SetsMixin:
    edges: list[Edge]
    nodes: dict[str, Node]
    orientovany: bool

    def get_mnozina_nasledniku(self, node: Node) -> set[Node]:
        if not self.orientovany:
            return set()

        mnozina = []

        for edge in self.edges:
            if edge.node1 == node and edge.direction == Direction.TO:
                mnozina.append(edge.node2)
            elif edge.node2 == node and edge.direction == Direction.FROM:
                mnozina.append(edge.node1)
        return set(mnozina)

    def get_mnozina_predchudcu(self, node: Node) -> set[Node]:
        if not self.orientovany:
            return set()

        mnozina = []

        for edge in self.edges:
            if edge.node1 == node and edge.direction == Direction.FROM:
                mnozina.append(edge.node2)
            elif edge.node2 == node and edge.direction == Direction.TO:
                mnozina.append(edge.node1)
        return set(mnozina)

    def get_mnozina_sousedu(self, node: Node) -> set[Node]:
        mnozina = []

        for edge in self.edges:
            if edge.node1 == node:
                mnozina.append(edge.node2)
            elif edge.node2 == node:
                mnozina.append(edge.node1)
        return set(mnozina)

    def get_vystupni_okoli(self, node: Node) -> list[Edge]:
        if not self.orientovany:
            return []

        mnozina = []

        for edge in self.edges:
            if edge.node1 == node and edge.direction == Direction.TO:
                mnozina.append(edge.name)
            elif edge.node2 == node and edge.direction == Direction.FROM:
                mnozina.append(edge.name)
        return mnozina

    def get_vstupni_okoli(self, node: Node) -> list[Edge]:
        if not self.orientovany:
            return []

        mnozina = []

        for edge in self.edges:
            if edge.node1 == node and edge.direction == Direction.FROM:
                mnozina.append(edge.name)
            elif edge.node2 == node and edge.direction == Direction.TO:
                mnozina.append(edge.name)
        return mnozina

    def get_okoli(self, node: Node) -> list[Edge]:
        mnozina = []

        for edge in self.edges:
            if edge.node1 == node or edge.node2 == node:
                mnozina.append(edge)
        return mnozina

    @property
    def tabulka_incidentnich_hran(self) -> dict[Any, Any]:
        tabulka = {"vystupni": {}, "vstupni": {}}
        for node in self.nodes.values():
            tabulka["vystupni"][node.name] = self.get_vystupni_okoli(node)
            tabulka["vstupni"][node.name] = self.get_vstupni_okoli(node)
        return tabulka

    @property
    def seznam_sousedu(self):
        seznam = []
        for node in self.nodes.values():
            seznam.append({node.name: self.get_mnozina_sousedu(node)})
        return seznam