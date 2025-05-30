from ds.trie import Trie

trie = Trie()
trie.insert("apple")
trie.insert("app")
trie.insert("apply")

print(trie.search("apple"))     # True
print(trie.search("app"))       # True
print(trie.search("appl"))      # False
print(trie.starts_with("app"))  # True
print(trie.starts_with("apl"))  # False

trie.delete("apple")
print(trie.search("apple"))     # False
print(trie.search("apply"))     # True
