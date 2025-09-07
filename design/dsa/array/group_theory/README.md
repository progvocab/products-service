 **Group Theory in Competitive Programming**.

---

# 🔹 What is Group Theory?

Group theory studies **algebraic structures** called **groups**, which capture symmetry and transformations.

A **group** is a set `G` with an operation `*` such that:

1. **Closure**: For any `a, b ∈ G`, `a * b ∈ G`.
2. **Associativity**: `(a * b) * c = a * (b * c)`.
3. **Identity**: There exists an element `e ∈ G` such that `a * e = e * a = a`.
4. **Inverse**: For each `a ∈ G`, there exists `a⁻¹` such that `a * a⁻¹ = a⁻¹ * a = e`.

---

## 🔹 Examples Relevant to Competitive Programming

### 1. **Permutations (Symmetric Group Sn)**

* The set of all permutations of `n` elements forms a group under composition.
* Operations like **rotate left** and **swap first/last** are just **permutations**.
* This is why earlier we could argue sorting was possible: because those two operations generate all permutations (`Sn`).

---

### 2. **Modular Arithmetic**

* `{0, 1, 2, …, n-1}` with addition mod `n` is a group.
* Useful in competitive programming for:

  * Hashing
  * Number theory problems
  * Cyclic sequences

---

### 3. **Matrix Groups**

* Set of invertible matrices under multiplication form a group.
* Used in **linear recurrences** (e.g., Fibonacci using matrix exponentiation).

---

---

# 🔹 Common Competitive Programming Problems Involving Groups

### **Problem 1: Modular Inverse**

Given `a` and `m` (coprime), find `x` such that:

```
(a * x) % m = 1
```

This is asking for the **inverse in the multiplicative group mod m**.

✅ Python code:

```python
def mod_inverse(a, m):
    # Extended Euclidean Algorithm
    def egcd(a, b):
        if b == 0:
            return a, 1, 0
        g, x1, y1 = egcd(b, a % b)
        return g, y1, x1 - (a // b) * y1

    g, x, _ = egcd(a, m)
    if g != 1:
        return None  # No inverse if gcd ≠ 1
    else:
        return x % m

print(mod_inverse(3, 11))  # Output: 4 (since 3*4 % 11 = 1)
```

---

### **Problem 2: Counting Distinct Rotations**

Given a string `s`, how many **distinct rotations** does it have?
This relates to the **cyclic group** of rotations.

✅ Python code:

```python
def distinct_rotations(s):
    n = len(s)
    rotations = {s[i:] + s[:i] for i in range(n)}
    return len(rotations)

print(distinct_rotations("abab"))  # Output: 2 ("abab", "baba")
```

---

### **Problem 3: Group Generator Problem**

If you have operations (like rotate, swap), can you generate the **entire permutation group Sn**?
This is exactly the sorting problem we discussed earlier ✅.

---

### **Problem 4: Fast Exponentiation in Groups**

Exponentiation in a group (like `a^k`) is common.
In modular arithmetic, this is **modular exponentiation**.

✅ Python code:

```python
def mod_pow(a, b, m):
    result = 1
    a = a % m
    while b > 0:
        if b & 1:
            result = (result * a) % m
        a = (a * a) % m
        b >>= 1
    return result

print(mod_pow(3, 200, 13))  # 3^200 % 13
```

---

### **Problem 5: Cayley’s Theorem (Permutation as a Group)**

Any group of size `n` can be seen as a subgroup of permutations (`Sn`).
Competitive problem example:

* You’re given operations (like swaps, rotations). Can you reach all permutations?

We already saw this with **rotate + swap**.

---

# 🔹 Why Group Theory is Useful in CP?

1. **Modular arithmetic** → number theory problems, cryptography.
2. **Permutations** → string/array manipulation, sorting by limited moves.
3. **Cycle decomposition** → used in analyzing permutation problems (like finding minimal swaps).
4. **Exponentiation in groups** → matrix exponentiation, modular exponentiation.
5. **Combinatorics** → Burnside’s Lemma, Polya’s Enumeration (advanced).

---

👉 Do you want me to **pick 2–3 real Codeforces/LeetCode style problems** and solve them step by step using group theory ideas (with Python code)? That way you’ll see how theory directly maps to practice.
