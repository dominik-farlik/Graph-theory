from collections import deque
import random
from typing import List, Optional

import numpy as np
from numpy._typing import NDArray

from edge.edge import Direction, Edge
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

    def minimalni_kostra(self) -> List[Edge]:
        """
        Vrátí seznam hran tvořících minimální kostru (minimum spanning tree).
        Pokud je graf nesouvislý, vrací minimální les (víc koster – pro každou komponentu).
        """

        node_list = list(self.nodes.values())
        n = len(node_list)

        if n <= 1:
            return []

        parent = {node: node for node in node_list}
        rank = {node: 0 for node in node_list}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False  # už ve stejné komponentě
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        edges_with_weights: list[tuple[float, Edge]] = []

        for e in self.edges:
            u, v = e.node1, e.node2

            # Smyčky pro MST ignorujeme
            if u == v:
                continue

            if e.length is None:
                w = 1.0
            else:
                w = float(e.length)

            edges_with_weights.append((w, e))

        if not edges_with_weights:
            return []

        # seřadit podle váhy
        edges_with_weights.sort(key=lambda item: item[0])

        mst_edges: List[Edge] = []

        for w, e in edges_with_weights:
            u, v = e.node1, e.node2
            if union(u, v):
                mst_edges.append(e)
                if len(mst_edges) == n - 1:
                    break

        return mst_edges

    def maximalni_kostra(self) -> List[Edge]:
        """
        Vrátí seznam hran tvořících maximální kostru (maximum spanning tree).
        U nesouvislého grafu vrací maximální les.
        """
        node_list = list(self.nodes.values())
        n = len(node_list)

        if n <= 1:
            return []

        # Union-Find
        parent = {node: node for node in node_list}
        rank = {node: 0 for node in node_list}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        # připravíme hrany + váhy
        edges_with_weights: list[tuple[float, Edge]] = []

        for e in self.edges:
            u, v = e.node1, e.node2
            if u == v:
                continue  # smyčky ignorujeme

            # případně můžeš filtrovat podle direction, viz min. kostra
            # if self.orientovany and e.direction != Direction.NONE:
            #     continue

            if e.length is None:
                w = 1.0
            else:
                w = float(e.length)

            edges_with_weights.append((w, e))

        if not edges_with_weights:
            return []

        # rozdíl oproti minimální kostře: reverse=True
        edges_with_weights.sort(key=lambda item: item[0], reverse=True)

        mst_edges: List[Edge] = []

        for w, e in edges_with_weights:
            u, v = e.node1, e.node2
            if union(u, v):
                mst_edges.append(e)
                if len(mst_edges) == n - 1:
                    break

        return mst_edges

    def _floyd_warshall(self) -> tuple[NDArray[np.float64], list[list[Optional[str]]]]:
        """
        Vypočítá matice nejkratších vzdáleností a předchůdců (Floyd–Warshall).
        Vrací (dist, pred), kde:
          dist[i, j] = délka nejkratší cesty i -> j
          pred[i][j] = jméno předchůdce vrcholu j na nejkratší cestě z i do j
        """
        node_list = list(self.nodes.values())
        n = len(node_list)

        dist = self.matice_delek.copy()
        pred: list[list[Optional[str]]] = [[None for _ in range(n)] for _ in range(n)]

        # inicializace předchůdců pro přímé hrany
        for i, u in enumerate(node_list):
            for j, v in enumerate(node_list):
                if i != j and not np.isinf(dist[i, j]):
                    pred[i][j] = u.name

        # Floyd–Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        pred[i][j] = pred[k][j]

        return dist, pred

    def nejkratsi_cesta(self, start_name: str, end_name: str) -> tuple[List[Node], float]:
        node_list = list(self.nodes.values())
        index_by_name = {node.name: i for i, node in enumerate(node_list)}

        if start_name not in index_by_name or end_name not in index_by_name:
            raise ValueError("Start nebo end uzel v grafu neexistuje.")

        i = index_by_name[start_name]
        j = index_by_name[end_name]

        D = self.matice_nejkratsich_delek  # POZOR: už ne matice_delek
        P = self.matice_predchudcu

        # neexistuje cesta
        if np.isinf(D[i, j]):
            return [], float('inf')

        # rekonstruuj cestu backtrackingem přes pred
        path_names = [end_name]
        curr_idx = j

        while curr_idx != i:
            pred_name = P[i][curr_idx]
            if pred_name is None:
                # bezpečnostní pojistka – nemělo by nastat, pokud D[i, j] není inf
                break
            path_names.append(pred_name)
            curr_idx = index_by_name[pred_name]

        path_names.reverse()
        path_nodes = [self.nodes[name] for name in path_names]
        length = float(D[i, j])

        return path_nodes, length
