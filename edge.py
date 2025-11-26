from enum import Enum

from node import Node


class Direction(Enum):
    TO = '>'
    FROM = '<'
    NONE = '-'


class Edge:
    def __init__(self, node1: Node, node2: Node, direction: Direction, name: str = None, length: float | None = None):
        if name:
            self.name = name
        else:
            self.name = node1.name + node2.name
        self.node1 = node1
        self.node2 = node2
        self.direction = direction
        self.length = length

    def __repr__(self):
        return "{name}({length}): {node1} {direction} {node2}".format(name=self.name, length=self.length, node1=self.node1, direction=self.direction.value, node2=self.node2)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False

        if self.direction == Direction.NONE and other.direction == Direction.NONE:
            return {self.node1, self.node2} == {other.node1, other.node2}

        return (
                self.node1 == other.node1 and
                self.node2 == other.node2 and
                self.direction == other.direction
        )

    def __hash__(self):
        if self.direction == Direction.NONE:
            return hash((frozenset({self.node1, self.node2}), self.direction))
        else:
            return hash((self.node1, self.node2, self.direction))
