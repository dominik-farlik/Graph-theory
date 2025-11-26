from node.node import Node


class DegreesMixin:
    def get_vystupni_stupen(self, node: Node) -> int:
        return len(self.get_vystupni_okoli(node))

    def get_vstupni_stupen(self, node: Node) -> int:
        return len(self.get_vstupni_okoli(node))

    def get_stupen(self, node: Node) -> int:
        return len(self.get_okoli(node))