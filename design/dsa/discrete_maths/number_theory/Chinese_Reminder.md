Nice — CRT is a lovely mix of algebra and algorithms. I’ll explain the theorem, why it works, how to handle both the **pairwise-coprime** and **general (possibly non-coprime)** cases, show a couple of proofs/intuition sketches, give practical algorithms used in competitive programming, and finish with robust Python code and examples you can drop into contests.

# What the Chinese Remainder Theorem (CRT) says — short version

Given a system of congruences

$$
x \equiv a_1 \pmod{n_1},\quad
x \equiv a_2 \pmod{n_2},\quad
\dots,\quad
x \equiv a_k \pmod{n_k},
$$

* If the moduli $n_i$ are **pairwise coprime** (i.e. $\gcd(n_i,n_j)=1$ for all $i\ne j$), then there exists a unique solution modulo

  $$
  N = n_1 n_2 \cdots n_k.
  $$

  That is, a solution $x$ exists and any two solutions are congruent mod $N$.

* If the moduli are **not** pairwise coprime, a solution exists **iff** every pair of congruences is compatible: for all $i,j$,

  $$
  a_i \equiv a_j \pmod{\gcd(n_i,n_j)}.
  $$

  If compatible, congruences can be merged to a single congruence modulo $\mathrm{lcm}(n_1,\dots,n_k)$.

---

# Intuition / proof sketch (pairwise coprime)

* Because moduli are pairwise coprime, by the Chinese remainder construction you can build a number that is `ai (mod ni)` and `0 (mod all other nj)`. Sum those terms and you get a number that satisfies every congruence.
* Uniqueness modulo $N$ follows because if two numbers agree modulo each $n_i$ then their difference is divisible by every $n_i$, hence by the product $N$.

Short constructive proof gives the algorithm below (product + modular inverses).

---

# Two practical algorithms

## 1) Pairwise-coprime CRT (fast direct formula)

Let $N = \prod n_i$. For each `i`:

* $N_i = N / n_i$.
* Find $y_i$ such that $N_i \cdot y_i \equiv 1 \pmod{n_i}$ (modular inverse).
  Then the solution is

$$
x \equiv \sum_i a_i \cdot N_i \cdot y_i \pmod{N}.
$$

This is the standard O(k log N) approach using modular inverses (via extended gcd).

## 2) General CRT (pairwise or not) — merge pairwise

Merge congruences two at a time:

* For two congruences $x \equiv a_1 \pmod{n_1}$ and $x \equiv a_2 \pmod{n_2}$:

  * Compute $g, p, q$ = `extended_gcd(n1, n2)` giving $p n_1 + q n_2 = g$.
  * If $(a_2 - a_1) \% g \ne 0$ → **no solution**.
  * Otherwise, there exists a solution modulo $\mathrm{lcm}(n_1, n_2) = n_1/g \cdot n_2$.
  * One concrete merged solution is:

    $$
    x = a_1 + n_1 \cdot \left( \left(\frac{a_2-a_1}{g} \cdot p\right) \bmod \frac{n_2}{g}\right).
    $$
  * Reduce `x mod lcm` and continue merging with the next congruence.

This method is robust and works for any moduli; it’s the method used in many CP implementations.

---

# Example (pairwise-coprime)

Solve:

$$
x \equiv 2 \pmod{3},\quad x \equiv 3 \pmod{5},\quad x \equiv 2 \pmod{7}.
$$

Using the direct formula (or just testing), solution is $x \equiv 23 \pmod{105}$ (since 23 %3 =2, %5 =3, %7 =2).

---

# Example (non-coprime)

Solve:

$$
x \equiv 2 \pmod{6},\quad x \equiv 8 \pmod{10}.
$$

Check gcd(6,10)=2. Requirement: $2 \equiv 8 \pmod{2}$ → yes (both even), so solutions exist. Merge to modulus lcm(6,10)=30. Merged solution computed via pairwise method yields some `x ≡ r (mod 30)`.

---

# Complexity

* Pairwise-coprime direct formula: O(k · log N) for inverse computations (log via extended gcd).
* Pairwise merging: O(k · log M) where M grows as lcm of processed moduli; each extended\_gcd is log-scale.

---

# Python: robust implementations

### helper: extended gcd and modinv

```python
def extended_gcd(a, b):
    # returns (g, x, y) where g = gcd(a,b) and a*x + b*y = g
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return (g, x, y)

def modinv(a, m):
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        return None  # inverse doesn't exist
    return x % m
```

### A. CRT for pairwise-coprime moduli

```python
def crt_pairwise(ai, ni):
    """
    ai: list of remainders
    ni: list of pairwise-coprime moduli
    returns (x, N) where x is solution modulo N = product(ni)
    """
    assert len(ai) == len(ni)
    N = 1
    for mod in ni:
        N *= mod

    x = 0
    for a, n in zip(ai, ni):
        Ni = N // n
        inv = modinv(Ni, n)
        if inv is None:
            raise ValueError("moduli must be pairwise coprime")
        x = (x + a * Ni * inv) % N

    return x % N, N
```

### B. General CRT (merge pairwise, works for non-coprime)

```python
def crt_general(ai, ni):
    """
    ai: list of remainders
    ni: list of moduli (not necessarily coprime)
    Returns (r, m) where r is solution modulo m (the merged modulus),
    or (None, None) if no solution.
    """
    assert len(ai) == len(ni)
    r, m = ai[0] % ni[0], ni[0]

    for a, n in zip(ai[1:], ni[1:]):
        g, p, q = extended_gcd(m, n)
        if (a - r) % g != 0:
            return None, None  # no solution
        # compute one solution
        lcm = m // g * n
        # multiplier to step r to match a modulo n
        mult = ((a - r) // g) * p
        mult %= (n // g)
        r = (r + m * mult) % lcm
        m = lcm

    return r % m, m
```

---

# Quick tests / usage

```python
# pairwise example
a = [2,3,2]
n = [3,5,7]
print(crt_pairwise(a,n))   # (23, 105)

# general example
a = [2, 8]
n = [6, 10]
print(crt_general(a,n))    # (... , 30) e.g. (8,30) or equivalent residue
# verify: 8 % 6 = 2, 8 % 10 = 8
```

(You can run these snippets directly — they’re standard contest-ready implementations.)

---

# Common CP uses of CRT

* Reconstructing an integer from remainders (e.g., when combining results mod different primes).
* Changing modulus in combinatorics: compute nCr mod composite by computing mod prime powers and combining.
* Speeding RSA/decryption using CRT (perform decryption mod p and q separately, then combine).
* Problems that ask to find smallest `x` satisfying multiple modular constraints (common in constructive/problem-solving tasks).

---

# Practical tips & pitfalls

* Always check pairwise coprimality when you assume the simple formula. If not coprime, use the merging method.
* Be careful with negative remainders; normalize remainders to `0..mod-1`.
* Watch overflow when multiplying moduli for very large lists — often you only need `r mod current_modulus` while merging, so avoid computing full product if not necessary.
* Use 64-bit integers if product fits; otherwise use Python big ints (automatic).

---

# Final compact example (full run)

Solve:

$$
x\equiv 2 \pmod{3},\quad x\equiv 3 \pmod{5},\quad x\equiv 2 \pmod{7}
$$

```python
a = [2,3,2]
n = [3,5,7]
r, M = crt_pairwise(a,n)
print(r, M)  # 23 105
# so x = 23 + 105*t
```

---

If you want I can:

* Walk this with a visual diagram (how each term Ni·yi kills other moduli), or
* Provide a version of CRT that outputs the smallest non-negative solution and verifies pairwise compatibility automatically, or
* Show how to combine CRT with fast exponentiation to speed up computations (e.g., modular exponent with composite modulus via prime-power decomposition).

Which follow-up would you like?
