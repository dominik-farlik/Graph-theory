import os
from pathlib import Path

import numpy as np
from numpy import ndarray

ROOT_DIR = Path(__file__).resolve().parent.parent
MATICE_DIR = os.path.join(ROOT_DIR, "matice")

os.makedirs(MATICE_DIR, exist_ok=True)

class ExportMixin:
    def print_matice(self, matice: ndarray, mocnina: int = 1):
        nodes = list(self.nodes.keys())
        with open(os.path.join(MATICE_DIR, f"matice_sousednosti_na_{mocnina}.csv"), "w", encoding="utf-8") as f:
            f.write("Node," + ",".join(nodes) + "\n")
            for i, row in enumerate(matice):
                f.write(f"{nodes[i]}," + ",".join(str(int(x)) for x in row) + "\n")

    def print_matice_incidence(self) -> None:
        B = self.matice_incidence
        node_names = [n.name for n in self.nodes.values()]
        col_names = [(e.name if e.name else f"h{j}") for j, e in enumerate(self.edges)]

        width = max(3, max(len(n) for n in node_names))
        colw = max(3, max(len(c) for c in col_names))
        header = "Nodes/Edges," + ",".join(c.rjust(colw) for c in col_names)

        with open(os.path.join(MATICE_DIR, "matice_incidence.csv"), "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for i, row in enumerate(B):
                f.write(node_names[i].rjust(width) + "," + ",".join(str(int(x)).rjust(colw) for x in row) + "\n")

    def print_znamenkova_matice(self):
        nodes = list(self.nodes.keys())
        with open(os.path.join(MATICE_DIR, "matice_znamenkova.csv"), "w", encoding="utf-8") as f:
            f.write("Node," + ",".join(nodes) + "\n")
            for i, row in enumerate(self.znamenkova_matice):
                f.write(f"{nodes[i]}," + ",".join((str(x)) for x in row) + "\n")

    def print_matice_delek(self) -> None:
        A = self.matice_delek
        nodes = [n.name for n in self.nodes.values()]
        with open(os.path.join(MATICE_DIR, "matice_delek.csv"), "w", encoding="utf-8") as file:
            file.write("Node," + ",".join(nodes) + "\n")
            for i, row in enumerate(A):
                def f(x): return "inf" if np.isinf(x) else (str(int(x)) if float(x).is_integer() else f"{x}")
                file.write(f"{nodes[i]}," + ",".join(f(x) for x in row) + "\n")

    def print_matice_predchudcu(self) -> None:
        P = self.matice_predchudcu

        nodes = [n.name for n in self.nodes.values()]
        with open(os.path.join(MATICE_DIR, "matice_predchudcu.csv"), "w", encoding="utf-8") as file:
            file.write("Node," + ",".join(nodes) + "\n")
            for i, row in enumerate(P):
                file.write(f"{nodes[i]}," + ",".join(x if x is not None else "-" for x in row) + "\n")

    def print_tabulka_incidentnich_hran(self) -> None:
        tab = self.tabulka_incidentnich_hran
        with open(os.path.join(MATICE_DIR, "tabulka_incidentnich_hran.csv"), "w", encoding="utf-8") as file:
            file.write("VÝSTUPNÍ HRANY:\n")
            for node, edges in tab["vystupni"].items():
                file.write(f"{node}," + ",".join(e for e in edges) + "\n")
            file.write("VSTUPNÍ HRANY:\n")
            for node, edges in tab["vstupni"].items():
                file.write(f"{node}," + ",".join(e for e in edges) + "\n")
