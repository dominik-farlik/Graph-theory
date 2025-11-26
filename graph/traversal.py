from typing import Any

from node.node import Node


class TraversalMixin:
    nodes: dict[str, Node]

    def get_root(self):
        for node, edges in self.tabulka_incidentnich_hran["vstupni"].items():
            if not edges:
                return self.nodes.get(node)

    def level_order(self):
        root = self.get_root()
        queue = [root]
        result = []
        while queue:
            result.append(queue[0])
            naslednici = self.get_mnozina_nasledniku(queue.pop(0))
            queue.extend(sorted(naslednici))

        return result

    def pre_order(self):
        pass

    def post_order(self):
        pass

