from collections import deque
import random

import numpy as np

from edge.edge import Direction
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
            print(queue[0])
            naslednici = self.get_mnozina_nasledniku(queue.pop(0))
            queue.extend(sorted(naslednici))

        return result

    def pre_order(self):
        root = self.get_root()
        result = []

        def dfs(node):
            result.append(node)
            for child in sorted(self.get_mnozina_nasledniku(node)):
                dfs(child)

        dfs(root)
        return result

    def post_order(self):
        root = self.get_root()
        result = []

        def dfs(node):
            for child in sorted(self.get_mnozina_nasledniku(node)):
                dfs(child)
            result.append(node)

        dfs(root)
        return result

    def in_order(self):
        root = self.get_root()
        result = []

        def dfs(node):
            children = sorted(self.get_mnozina_nasledniku(node))
            left = children[0] if len(children) > 0 else None
            right = children[1] if len(children) > 1 else None
            if left:
                dfs(left)
            result.append(node)
            if right:
                dfs(right)

        dfs(root)
        return result

    def bfs(self, start: Node | str | None = None) -> list[Node]:
        """Průchod grafem do šířky (BFS) od zadaného startu."""
        if start is None:
            start_node = random.choice(list(self.nodes.values()))
        elif isinstance(start, str):
            start_node = self.nodes[start]
        else:
            start_node = start

        if start_node is None:
            return []

        visited: set[Node] = set()
        order: list[Node] = []

        queue = deque([start_node])
        visited.add(start_node)

        while queue:
            node = queue.popleft()
            order.append(node)

            for neigh in sorted(self.get_mnozina_nasledniku(node), key=lambda n: n.name):
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)

        return order

    def dfs(self, start: Node | str | None = None) -> list[Node]:
        """Průchod grafem do hloubky (DFS, pre-order) od zadaného startu."""
        if start is None:
            start_node = random.choice(list(self.nodes.values()))
        elif isinstance(start, str):
            start_node = self.nodes[start]
        else:
            start_node = start

        if start_node is None:
            return []

        visited: set[Node] = set()
        order: list[Node] = []

        stack: list[Node] = [start_node]

        while stack:
            node = stack.pop()

            if node in visited:
                continue

            visited.add(node)
            order.append(node)

            neighbours = sorted(
                self.get_mnozina_nasledniku(node),
                key=lambda n: n.name,
                reverse=True,
            )
            for neigh in neighbours:
                if neigh not in visited:
                    stack.append(neigh)

        return order

    def pocet_koster(self) -> int:
        """
        Vrátí počet koster (spanning trees) neorientovaného grafu pomocí
        Kirchhoffovy věty. Pokud je graf nesouvislý, vrací 0.
        """
        node_list = list(self.nodes.values())
        n = len(node_list)

        if n == 0:
            return 0
        if n == 1:
            return 1  # jediný vrchol => 1 triviální kostra

        # mapování vrchol -> index
        index = {node: i for i, node in enumerate(node_list)}

        # Matice sousednosti A
        A = np.zeros((n, n), dtype=int)

        for e in self.edges:
            u, v = e.node1, e.node2
            i, j = index[u], index[v]

            # Pro kostry bereme neorientovaný graf.
            # Pokud máš orientovaný graf, buď ignoruj orientaci, nebo filtruj podle direction.
            if self.orientovany and e.direction != Direction.NONE:
                # IGNOROVAT čistě orientované hrany:
                # continue
                # NEBO zacházet s nimi jako s neorientovanými:
                pass

            if i == j:
                # smyčky ignorujeme – neovlivňují počet koster
                continue

            A[i, j] = 1
            A[j, i] = 1

        # Matice stupňů D
        degrees = A.sum(axis=1)
        D = np.diag(degrees)

        # Laplaceova matice L = D - A
        L = D - A

        # Kofaktor: smažeme 0. řádek a 0. sloupec (můžeš smazat libovolný)
        L_minor = L[1:, 1:]

        # Determinant – může vyjít např. 2.9999999997 kvůli numerice,
        # proto to zaokrouhlíme.
        det_val = np.linalg.det(L_minor)
        count = int(round(det_val))

        # pro jistotu nuluje velmi malé záporné zbytky typu -1e-9
        if count < 0 and abs(det_val) < 1e-6:
            count = 0

        return count
