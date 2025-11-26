from collections import deque


class BSTNode:
    def __init__(self, key: int):
        self.key = key
        self.left: "BSTNode | None" = None
        self.right: "BSTNode | None" = None

    def __repr__(self):
        return f"{self.key}"


class BST:
    def __init__(self):
        self.root: BSTNode | None = None

    def insert(self, key: int) -> None:
        if self.root is None:
            self.root = BSTNode(key)
            return

        current = self.root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = BSTNode(key)
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = BSTNode(key)
                    return
                current = current.right

    def delete(self, key: int) -> None:
        """Smaže jeden výskyt hodnoty `key` z BST (pokud existuje)."""
        self.root = self._delete_rec(self.root, key)

    def _delete_rec(self, node: BSTNode | None, key: int) -> BSTNode | None:
        if node is None:
            return None

        if key < node.key:
            node.left = self._delete_rec(node.left, key)
            return node

        if key > node.key:
            node.right = self._delete_rec(node.right, key)
            return node

        if node.left is None and node.right is None:
            return None

        if node.left is None:
            return node.right

        if node.right is None:
            return node.left

        succ = node.right
        while succ.left is not None:
            succ = succ.left

        node.key = succ.key
        node.right = self._delete_rec(node.right, succ.key)
        return node

    def pre_order(self) -> list[int]:
        result: list[int] = []

        def dfs(node: BSTNode | None):
            if node is None:
                return
            result.append(node.key)
            dfs(node.left)
            dfs(node.right)

        dfs(self.root)
        return result

    def in_order(self) -> list[int]:
        result: list[int] = []

        def dfs(node: BSTNode | None):
            if node is None:
                return
            dfs(node.left)
            result.append(node.key)
            dfs(node.right)

        dfs(self.root)
        return result

    def post_order(self) -> list[int]:
        result: list[int] = []

        def dfs(node: BSTNode | None):
            if node is None:
                return
            dfs(node.left)
            dfs(node.right)
            result.append(node.key)

        dfs(self.root)
        return result

    def level_order(self) -> list[int]:
        result: list[int] = []
        if self.root is None:
            return result

        queue = deque([self.root])

        while queue:
            node = queue.popleft()
            result.append(node.key)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result

    def pretty_print(self) -> None:
        """Vytiskne strom v textové podobě."""

        def _print(node: BSTNode | None, prefix: str = "", is_left: bool = True):
            if node is None:
                return

            # samotný uzel
            konektor = "└── " if is_left else "┌── "
            print(prefix + konektor + str(node.key))

            # prefix pro potomky
            if is_left:
                new_prefix = prefix + "    "
            else:
                new_prefix = prefix + "│   "

            # pravý pak levý, aby to vypadalo trochu jako strom
            _print(node.right, new_prefix, False)
            _print(node.left, new_prefix, True)

        if self.root is None:
            print("(prázdný strom)")
        else:
            _print(self.root)


def build_bst_from_file(path: str) -> BST:
    bst = BST()
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    numbers = [int(part.strip()) for part in content.split(",") if part.strip()]
    for num in numbers:
        bst.insert(num)

    return bst
