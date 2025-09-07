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


Burnside's Lemma, also known as the Orbit-Stabilizer Theorem or the Orbit Counting Theorem, provides a formula for counting the number of distinct configurations (orbits) under a group action. 
Formula: 
The number of orbits, denoted as $|X/G|$, is given by: 
$|X/G| = \frac{1}{|G|} \sum_{g \in G} |X^g|$ 
Where: 

• $|X/G|$ is the number of distinct orbits. 
• $G$ is the group acting on the set $X$. 
• $|G|$ is the order of the group (the number of elements in the group). 
• $X^g$ is the set of elements in $X$ that are fixed by the permutation $g \in G$ (i.e., $x \in X^g$ if $g(x) = x$). 
• $|X^g|$ is the number of fixed points of $g$. 

Python Code Example (for counting distinct necklaces): 
This example demonstrates the application of Burnside's Lemma to count the number of distinct necklaces of length n using k colors, where rotations are the only allowed group operations. 
```python 

from math import gcd

def count_distinct_necklaces(n, k):
    """
    Counts the number of distinct necklaces of length n using k colors
    under rotational symmetry using Burnside's Lemma.
    """
    
    total_fixed_points = 0
    # Iterate through all possible rotations (elements of the group)
    for i in range(n):
        # The number of fixed points for a rotation by 'i' steps
        # is k raised to the power of gcd(i, n).
        # This is because elements are fixed if they are constant within cycles
        # formed by the rotation, and the length of these cycles is determined by gcd(i, n).
        total_fixed_points += k ** gcd(i, n)
    
    # Divide by the order of the group (n, for rotations)
    num_distinct_necklaces = total_fixed_points // n
    return num_distinct_necklaces

# Example Usage:
n_stones = 4  # Length of the necklace (number of stones)
m_colors = 3  # Number of available colors

result = count_distinct_necklaces(n_stones, m_colors)
print(f"Number of distinct necklaces with {n_stones} stones and {m_colors} colors: {result}")
```
AI responses may include mistakes.

