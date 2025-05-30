# trie_module.py

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.end_of_word = True

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.end_of_word

    def starts_with(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None

    def delete(self, word: str) -> bool:
        return self._delete(self.root, word, 0)

    def _delete(self, node: TrieNode, word: str, index: int) -> bool:
        if index == len(word):
            if not node.end_of_word:
                return False  # Word not found
            node.end_of_word = False
            return len(node.children) == 0  # Can delete node if it's a leaf
        ch = word[index]
        if ch not in node.children:
            return False  # Word not found
