import os

from graph import Graph

TYPE = 0

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


if __name__ == "__main__":
    if not os.path.exists("./matice"):
        os.makedirs("./matice")

    G = load_graph_from_file("graphs/09.tg")
    G.plot()
    G.print_properties()

    print("Zadej uzel:")
    node = G.nodes.get("", None) # node to print its properties
    if node:
        print(f"Množina následníků uzlu {node}: {G.get_mnozina_nasledniku(node)}")
        print(f"Množina předchůdců uzlu {node}: {G.get_mnozina_predchudcu(node)}")
        print(f"Množina sousedů uzlu {node}: {G.get_mnozina_sousedu(node)}")
        print(f"Výstupní okolí uzlu {node}: {G.get_vystupni_okoli(node)}")
        print(f"Vstupní okolí uzlu() {node}: {G.get_vstupni_okoli(node)}")
        print(f"Okolí uzlu {node}: {G.get_okoli(node)}")
        print(f"Výstupní stupeň uzlu {node}: {G.get_vystupni_stupen(node)}")
        print(f"Vstupní stupeň uzlu {node}: {G.get_vstupni_stupen(node)}")
        print(f"Stupeň uzlu {node}: {G.get_stupen(node)}")

    # matice
    G.print_matice(G.matice_sousednosti)
    G.print_matice(G.get_mocnina_matice_sousednosti(8), 8)
    G.print_znamenkova_matice()
    G.print_matice_incidence()
    G.print_matice_delek()
    G.print_matice_predchudcu()
    G.print_tabulka_incidentnich_hran()

    # seznamy
    print("seznam sousedu:\n", G.seznam_sousedu)
    print("seznam uzlu:\n", list(G.nodes.keys()))
    print("seznam hran:\n" + ", ".join(e.name for e in G.edges))

    # kostry
    print("level order:\n", G.level_order)

