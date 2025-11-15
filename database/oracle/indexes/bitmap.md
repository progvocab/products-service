A **bitmap** (or bitset) is a data structure that compactly stores bits (0s and 1s), often used to represent a set of integers or flags in a memory-efficient way.

---

## **Bitmap Use Cases**
- used in OLAP,  not suitable for OLTP as it locks the entire range.
- Representing presence/absence (e.g., bloom filters, indexing in DBs like Oracle)
- Set operations (union, intersection)
- Sparse data compression

---

## **Python Implementation of a Simple Bitmap**

```python
class Bitmap:
    def __init__(self, size):
        self.size = size
        self.bitmap = [0] * ((size // 32) + 1)  # 32 bits per int

    def set_bit(self, num):
        index = num // 32
        pos = num % 32
        self.bitmap[index] |= 1 << pos

    def clear_bit(self, num):
        index = num // 32
        pos = num % 32
        self.bitmap[index] &= ~(1 << pos)

    def check_bit(self, num):
        index = num // 32
        pos = num % 32
        return (self.bitmap[index] >> pos) & 1

# Example usage:
bm = Bitmap(100)
bm.set_bit(10)
bm.set_bit(65)

print(bm.check_bit(10))  # Output: 1
print(bm.check_bit(65))  # Output: 1
print(bm.check_bit(33))  # Output: 0

bm.clear_bit(10)
print(bm.check_bit(10))  # Output: 0
```

---

## **Bitmap Operations**

```python
# Union of two bitmaps
def union(bm1, bm2):
    return [a | b for a, b in zip(bm1, bm2)]

# Intersection of two bitmaps
def intersection(bm1, bm2):
    return [a & b for a, b in zip(bm1, bm2)]
```

---

## **Advantages**
- Very memory efficient.
- Fast bitwise operations (useful for filtering, set ops).
- Perfect for low cardinality sets (as used in Oracle bitmap indexes).

---

Want to see a real-world use case like bitmap-based filtering in analytics or bloom filters?