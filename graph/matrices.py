from typing import Optional

import numpy as np
from numpy import ndarray
from numpy._typing import NDArray

from edge.edge import Direction, Edge
from node.node import Node


class MatrixMixin:
    edges: list[Edge]
    nodes: dict[str, Node]
    orientovany: bool

    @property
    def matice_sousednosti(self) -> ndarray:
        n = len(self.nodes)
        node_list = list(self.nodes.values())
        index = {node: i for i, node in enumerate(node_list)}
        A = np.zeros((n, n), dtype=float)

        for edge in self.edges:
            u, v = edge.node1, edge.node2
            i, j = index[u], index[v]
            A[i, j] = 1
            A[j, i] = 1

        return A

    def get_mocnina_matice_sousednosti(self, mocnina: int = 2) -> ndarray:
        return np.linalg.matrix_power(self.matice_sousednosti, mocnina)

    @property
    def matice_incidence(self) -> ndarray:
        node_list = list(self.nodes.values())
        index = {node: i for i, node in enumerate(node_list)}
        n = len(node_list)
        m = len(self.edges)

        B = np.zeros((n, m), dtype=int)

        for j, e in enumerate(self.edges):
            u, v = e.node1, e.node2
            iu, iv = index[u], index[v]

            if not self.orientovany or e.direction == Direction.NONE:
                if u == v:
                    B[iu, j] = 2
                else:
                    B[iu, j] = 1
                    B[iv, j] = 1
            else:
                if u == v:
                    B[iu, j] = 2
                elif e.direction == Direction.TO:
                    B[iu, j] = +1
                    B[iv, j] = -1
                elif e.direction == Direction.FROM:
                    B[iv, j] = +1
                    B[iu, j] = -1
                else:
                    B[iu, j] = 1
                    B[iv, j] = 1

        return B



    @property
    def znamenkova_matice(self) -> list[list[str]]:
        if self.orientovany:
            node_list = list(self.nodes.values())
            index = {node: i for i, node in enumerate(node_list)}
            A = []

            for i, row in enumerate(node_list):
                A.append([])
                for j, col in enumerate(node_list):
                    if row == col:
                        A[i].append(0)
                    else:
                        A[i].append('-')

            for edge in self.edges:
                u, v = edge.node1, edge.node2
                i, j = index[u], index[v]
                if edge.direction == Direction.TO:
                    A[i][j] = '+'
                else:
                    A[j][i] = '+'

            return A
        return []

    @property
    def matice_delek(self) -> ndarray:
        node_list = list(self.nodes.values())
        index = {node: i for i, node in enumerate(node_list)}
        n = len(node_list)
        A = np.full((n, n), np.inf, dtype=float)
        np.fill_diagonal(A, 0.0)

        for e in self.edges:
            u, v = e.node1, e.node2
            i, j = index[u], index[v]
            w = 1.0 if e.length is None else float(e.length)

            if not self.orientovany or e.direction == Direction.NONE:
                if i == j:
                    A[i, i] = min(A[i, i], w)
                else:
                    A[i, j] = min(A[i, j], w)
                    A[j, i] = min(A[j, i], w)
            else:
                if e.direction == Direction.TO:
                    A[i, j] = min(A[i, j], w)
                elif e.direction == Direction.FROM:
                    A[j, i] = min(A[j, i], w)

        return A

    @property
    def matice_nejkratsich_delek(self) -> NDArray[np.float64]:
        dist, _ = self._floyd_warshall()
        return dist

    @property
    def matice_predchudcu(self) -> list[list[Optional[str]]]:
        _, pred = self._floyd_warshall()
        return pred
