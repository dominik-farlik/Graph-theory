import os

from graph.core import load_graph_from_file
from BST.BST import build_bst_from_file

def print_node_properties(node):
    if not node:
        print("Nebyl zadán uzel pro analýzu")
        return
    print(f"Množina následníků uzlu {node}: {G.get_mnozina_nasledniku(node)}")
    print(f"Množina předchůdců uzlu {node}: {G.get_mnozina_predchudcu(node)}")
    print(f"Množina sousedů uzlu {node}: {G.get_mnozina_sousedu(node)}")
    print(f"Výstupní okolí uzlu {node}: {G.get_vystupni_okoli(node)}")
    print(f"Vstupní okolí uzlu() {node}: {G.get_vstupni_okoli(node)}")
    print(f"Okolí uzlu {node}: {G.get_okoli(node)}")
    print(f"Výstupní stupeň uzlu {node}: {G.get_vystupni_stupen(node)}")
    print(f"Vstupní stupeň uzlu {node}: {G.get_vstupni_stupen(node)}")
    print(f"Stupeň uzlu {node}: {G.get_stupen(node)}")

def export_matice():
    G.print_matice(G.matice_sousednosti)
    G.print_matice(G.get_mocnina_matice_sousednosti(2), 2)
    G.print_znamenkova_matice()
    G.print_matice_incidence()
    G.print_matice_delek()
    G.print_matice_predchudcu()
    G.print_tabulka_incidentnich_hran()

def print_seznamy():
    print("seznam sousedu: ", G.seznam_sousedu)
    print("seznam uzlu: ", list(G.nodes.keys()))
    print("seznam hran: " + ", ".join(e.name for e in G.edges))

def print_kostry():
    print("Pocet koster: ", G.pocet_koster())
    print("Minimalni kostra: ", G.minimalni_kostra())
    print("Hodnota minimální kostry: ", sum((float(e.length) or 1.0) for e in G.minimalni_kostra()))
    print("Maximální kostra: ", G.maximalni_kostra())
    print("Hodnota maximální kostry:", sum(float(e.length) if e.length is not None else 1.0 for e in G.maximalni_kostra()))

def print_graph_orders():
    #print("level order: ", G.level_order())
    #print("pre order: ", G.pre_order())
    #print("pre order: ", G.pre_order())
    #print("post order: ", G.post_order())
    #print("in order(nepozná pravá/levá u jednoho potomka): ", G.in_order())
    return

def print_prohledavani():
    print("Do hloubky: ", G.bfs())
    print("Do sirky: ", G.dfs())

    start, end = "s", "t"

    cesta, delka = G.nejkratsi_cesta(start, end)
    if cesta:
        print("Nejkratší cesta:", " -> ".join(node.name for node in cesta))
        print("Délka:", delka)
    else:
        print(f"Mezi {start} a {end} neexistuje žádná cesta.")

    cesta, delka = G.nejdelsi_cesta(start, end)
    if cesta:
        print("Nejdelší cesta:", " -> ".join(node.name for node in cesta))
        print("Délka:", delka)
    else:
        print(f"Mezi {start} a {end} neexistuje žádná cesta.")

    path, width = G.nejsirsi_cesta(start, end)
    if path:
        print("Nejširší cesta:", " -> ".join(node.name for node in path))
        print("Šířka:", width)
    else:
        print(f"Mezi {start} a {end} neexistuje žádná cesta.")

    #cesta, bezpecnost = G.nejbezpecnejsi_cesta(start, end)
    #print("Nejbezpečnější cesta:", " -> ".join(node.name for node in cesta))
    #print("Celková bezpečnost:", bezpecnost)

    print(f"Maximální tok ze {start} do {end}:", G.maximalni_tok(start, end))

def print_bst():
    print("Root:", BST.root)
    print("BST:", BST.level_order())
    BST.delete(53)
    BST.delete(12)
    BST.delete(77)
    print("BST:", BST.level_order())
    # bst.pretty_print()

if __name__ == "__main__":
    if not os.path.exists("./matice"):
        os.makedirs("./matice")

    G = load_graph_from_file("graphs/23.tg")
    #G.draw()
    G.print_properties()
    print_node_properties(G.nodes.get("", None))
    export_matice()
    print_seznamy()
    print_kostry()
    print_graph_orders()
    print_prohledavani()

    BST = build_bst_from_file("graphs/bst.txt")
    print_bst()
