Perfect 🚀 Let’s go step by step and cover **all the classic combinatorics operations** with Python’s `itertools` and `math`.

We’ll see:

1. **Factorial**
2. **Permutations (without repetition)**
3. **Combinations (without repetition)**
4. **Combinations with repetition**
5. **Permutations with repetition** ($n^r$)

---

## 🔹 Python Code

```python
import itertools
import math

items = ['A', 'B', 'C']   # n = 3
r = 2

# 1. Factorial
print("Factorial of n=3:", math.factorial(len(items)))  # 3! = 6

# 2. Permutations (without repetition)
print("\nPermutations (nPr):")
for p in itertools.permutations(items, r):
    print(p)
print("Count:", math.perm(len(items), r))  # 3P2 = 6

# 3. Combinations (without repetition)
print("\nCombinations (nCr):")
for c in itertools.combinations(items, r):
    print(c)
print("Count:", math.comb(len(items), r))  # 3C2 = 3

# 4. Combinations with repetition
print("\nCombinations with repetition (nHr):")
for cr in itertools.combinations_with_replacement(items, r):
    print(cr)
# Formula: C(n+r-1, r)
print("Count:", math.comb(len(items)+r-1, r))  # 3H2 = 6

# 5. Permutations with repetition (n^r)
print("\nPermutations with repetition (n^r):")
for pr in itertools.product(items, repeat=r):
    print(pr)
print("Count:", len(items) ** r)  # 3^2 = 9
```

---

## 🔹 Output (with `items = ['A','B','C'], r = 2`)

```
Factorial of n=3: 6

Permutations (nPr):
('A', 'B')
('A', 'C')
('B', 'A')
('B', 'C')
('C', 'A')
('C', 'B')
Count: 6

Combinations (nCr):
('A', 'B')
('A', 'C')
('B', 'C')
Count: 3

Combinations with repetition (nHr):
('A', 'A')
('A', 'B')
('A', 'C')
('B', 'B')
('B', 'C')
('C', 'C')
Count: 6

Permutations with repetition (n^r):
('A', 'A')
('A', 'B')
('A', 'C')
('B', 'A')
('B', 'B')
('B', 'C')
('C', 'A')
('C', 'B')
('C', 'C')
Count: 9
```

---

✅ This shows **all possible combinatorial cases** using Python.

Do you want me to also make a **visual table** comparing formulas, names, and Python equivalents for all 5 cases?
