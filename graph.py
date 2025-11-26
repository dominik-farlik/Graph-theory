from typing import Optional, Any, Dict, List, Set, Tuple, Callable, Union
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from node import Node
from edge import Edge, Direction


NODE_NAME = 1
NODE_1 = 1
DIRECTION = 2
NODE_2 = 3


class Graph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.edge_name = None
        self.edge_length = None
        self.edge_structure = False

    def __repr__(self):
        return "\n".join(repr(edge) for edge in self.edges)

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

    def print_matice(self, matice: ndarray, mocnina: int = 1):
        nodes = list(self.nodes.keys())
        with open(f"./matice/matice_sousednosti_na_{mocnina}.csv", "w", encoding="utf-8") as f:
            f.write("Node," + ",".join(nodes) + "\n")
            #print("   " + "  ".join(nodes))
            for i, row in enumerate(matice):
                f.write(f"{nodes[i]}," + ",".join(str(int(x)) for x in row) + "\n")
                #print(f"{nodes[i]}: " + "  ".join(str(int(x)) for x in row))

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

    def print_matice_incidence(self) -> None:
        B = self.matice_incidence
        node_names = [n.name for n in self.nodes.values()]
        col_names = [(e.name if e.name else f"h{j}") for j, e in enumerate(self.edges)]

        width = max(3, max(len(n) for n in node_names))
        colw = max(3, max(len(c) for c in col_names))
        header = "Nodes/Edges," + ",".join(c.rjust(colw) for c in col_names)

        with open("./matice/matice_incidence.csv", "w", encoding="utf-8") as f:
            f.write(header + "\n")
            #print(header)
            for i, row in enumerate(B):
                f.write(node_names[i].rjust(width) + "," + ",".join(str(int(x)).rjust(colw) for x in row) + "\n")
                #print(node_names[i].rjust(width) + ": " +" ".join(str(int(x)).rjust(colw) for x in row))

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

    def print_znamenkova_matice(self):
        nodes = list(self.nodes.keys())
        with open("./matice/matice_znamenkova.csv", "w", encoding="utf-8") as f:
            f.write("Node," + ",".join(nodes) + "\n")
            #print("   " + "  ".join(nodes))
            for i, row in enumerate(self.znamenkova_matice):
                f.write(f"{nodes[i]}," + ",".join((str(x)) for x in row) + "\n")
                #print(f"{nodes[i]}: " + "  ".join((str(x)) for x in row))

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
            w = 1.0 if e.length is None else e.length

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

    def print_matice_delek(self) -> None:
        A = self.matice_delek
        nodes = [n.name for n in self.nodes.values()]
        with open("./matice/matice_delek.csv", "w", encoding="utf-8") as file:
            file.write("Node," + ",".join(nodes) + "\n")
            #print("   " + "  ".join(nodes))
            for i, row in enumerate(A):
                def f(x): return "inf" if np.isinf(x) else (str(int(x)) if float(x).is_integer() else f"{x}")
                file.write(f"{nodes[i]}," + ",".join(f(x) for x in row) + "\n")
                #print(f"{nodes[i]}: " + "  ".join(f(x) for x in row))

    @property
    def matice_predchudcu(self) -> list[list[Optional[str]]]:
        node_list = list(self.nodes.values())
        n = len(node_list)
        dist = self.matice_delek.copy()
        pred = [[None for _ in range(n)] for _ in range(n)]

        for i, u in enumerate(node_list):
            for j, v in enumerate(node_list):
                if i != j and not np.isinf(dist[i, j]):
                    pred[i][j] = u.name

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        pred[i][j] = pred[k][j]

        return pred

    def print_matice_predchudcu(self) -> None:
        P = self.matice_predchudcu

        nodes = [n.name for n in self.nodes.values()]
        with open("./matice/matice_predchudcu.csv", "w", encoding="utf-8") as file:
            file.write("Node," + ",".join(nodes) + "\n")
            for i, row in enumerate(P):
                file.write(f"{nodes[i]}," + ",".join(x if x is not None else "-" for x in row) + "\n")

    @property
    def tabulka_incidentnich_hran(self) -> dict[Any, Any]:
        tabulka = {"vystupni": {}, "vstupni": {}}
        for node in self.nodes.values():
            tabulka["vystupni"][node.name] = self.get_vystupni_okoli(node)
            tabulka["vstupni"][node.name] = self.get_vstupni_okoli(node)
        return tabulka

    def print_tabulka_incidentnich_hran(self) -> None:
        tab = self.tabulka_incidentnich_hran
        print("Tabulka incidentních hran:")
        print("VÝSTUPNÍ HRANY:")
        for node, edges in tab["vystupni"].items():
            print(f"  {node}: " + ", ".join(e for e in edges))
        print("\nVSTUPNÍ HRANY:")
        for node, edges in tab["vstupni"].items():
            print(f"  {node}: " + ", ".join(e for e in edges))

    @property
    def seznam_sousedu(self):
        seznam = []
        for node in self.nodes.values():
            seznam.append({node.name: self.get_mnozina_sousedu(node)})
        return seznam

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

    def get_vystupni_stupen(self, node: Node) -> int:
        return len(self.get_vystupni_okoli(node))

    def get_vstupni_stupen(self, node: Node) -> int:
        return len(self.get_vstupni_okoli(node))

    def get_stupen(self, node: Node) -> int:
        return len(self.get_okoli(node))

    def plot(self):
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

    def get_root(self):
        for node, edges in self.tabulka_incidentnich_hran["vstupni"].items():
            if not edges:
                return self.nodes.get(node)

    @property
    def level_order(self):
        root = self.get_root()
        queue = [root]
        result = []
        while queue:
            result.append(queue[0])
            naslednici = self.get_mnozina_nasledniku(queue.pop(0))
            queue.extend(sorted(naslednici))

        return result
