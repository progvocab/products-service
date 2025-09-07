walk through **4 classic competitive-style problems** that directly use discrete-math ideas. For each: a short problem statement, the math idea, an algorithm (with complexity), and ready-to-run Python code.

---

# 1) Compute `nCr % p` for large `n` (prime `p`)

**Problem:** Given `n, r` up to e.g. `10^6` and prime `p`, compute `C(n, r) % p`.

**Math idea:**
`C(n,r) = n! / (r!(n-r)!)`. Under mod prime `p` we use modular inverses:
`C(n,r) % p = fact[n] * inv(fact[r]) * inv(fact[n-r]) % p`.
If `p` is prime, `inv(x) = x^(p-2) mod p` (Fermat). Precompute factorials and inverse factorials.

**Complexity:** Precompute O(n) + queries O(1).

```python
MOD = 10**9+7  # example prime

def precompute_factorials(n, mod=MOD):
    fact = [1] * (n+1)
    for i in range(1, n+1):
        fact[i] = fact[i-1] * i % mod
    inv_fact = [1] * (n+1)
    inv_fact[n] = pow(fact[n], mod-2, mod)  # Fermat inverse
    for i in range(n, 0, -1):
        inv_fact[i-1] = inv_fact[i] * i % mod
    return fact, inv_fact

def nCr_mod(n, r, fact, inv_fact, mod=MOD):
    if r < 0 or r > n: return 0
    return fact[n] * inv_fact[r] % mod * inv_fact[n-r] % mod

# Example
NMAX = 10**6
fact, inv_fact = precompute_factorials(NMAX)
print(nCr_mod(10, 3, fact, inv_fact))  # 120
```

---

# 2) Count distinct necklaces under rotation (Burnside’s Lemma)

**Problem:** Given `n` beads and `k` colors, count distinct necklaces up to rotation.

**Math idea:** Burnside’s Lemma (a.k.a. orbit-counting): number of distinct colorings =
`(1/|G|) * Σ_{g∈G} fix(g)`, where `G` is rotation group of size `n`. For a rotation by `d` positions, the number of colorings fixed is `k^{gcd(n, d)}`. So answer = `(1/n) * Σ_{d=0..n-1} k^{gcd(n,d)}`.

**Complexity:** O(n · log MOD) for pow if mod used.

```python
from math import gcd

def distinct_necklaces(n, k, mod=None):
    total = 0
    for d in range(n):
        cycles = gcd(n, d)
        val = pow(k, cycles, mod) if mod else k**cycles
        total += val
        if mod: total %= mod
    if mod:
        # divide by n mod mod -> multiply by inverse
        inv_n = pow(n, mod-2, mod)
        return total * inv_n % mod
    else:
        return total // n

# Examples
print(distinct_necklaces(4, 2))         # 6 (no mod)
print(distinct_necklaces(4, 2, 10**9+7))# with mod
```

---

# 3) `n`-th Fibonacci fast via matrix exponentiation

**Problem:** Compute `F(n)` where `F(0)=0, F(1)=1` for large `n` (e.g., `n` up to `10^18`).

**Math idea:** Linear recurrence solved by powers of transformation matrix:

```
[ F(n+1) ] = [1 1]^n [F(1)]
[ F(n)   ]   [1 0]   [F(0)]
```

So `F(n)` is entry `(0,1)` of `M^(n-1)` or use exponentiation by squaring.

**Complexity:** `O(log n)` matrix multiplications (2x2).

```python
def mat_mult(A, B, mod=None):
    a,b,c,d = A[0][0], A[0][1], A[1][0], A[1][1]
    e,f,g,h = B[0][0], B[0][1], B[1][0], B[1][1]
    if mod:
        return [[(a*e + b*g) % mod, (a*f + b*h) % mod],
                [(c*e + d*g) % mod, (c*f + d*h) % mod]]
    else:
        return [[a*e + b*g, a*f + b*h],
                [c*e + d*g, c*f + d*h]]

def mat_pow(M, n, mod=None):
    # exponentiate 2x2 matrix M^n
    res = [[1,0],[0,1]]
    base = M
    while n > 0:
        if n & 1:
            res = mat_mult(res, base, mod)
        base = mat_mult(base, base, mod)
        n >>= 1
    return res

def fib(n, mod=None):
    if n == 0: return 0
    M = [[1,1],[1,0]]
    P = mat_pow(M, n-1, mod)
    return P[0][0] if not mod else P[0][0] % mod

print(fib(10))           # 55
print(fib(10, 10**9+7))  # 55 mod
```

---

# 4) Minimum swaps to sort an array (permutation cycles)

**Problem:** Given a permutation represented as array `A` (distinct elements), find minimum number of swaps to sort it (swap any two elements).

**Math idea:** Represent permutation as cycles. Each cycle of length `L` needs `L-1` swaps. Sum over cycles.

**Complexity:** O(n).

```python
def min_swaps_to_sort(arr):
    n = len(arr)
    # pair value with original index, sort by value
    a = sorted((val, i) for i, val in enumerate(arr))
    visited = [False]*n
    ans = 0
    for i in range(n):
        if visited[i] or a[i][1] == i:
            continue
        cycle_size = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = a[j][1]
            cycle_size += 1
        if cycle_size > 0:
            ans += cycle_size - 1
    return ans

print(min_swaps_to_sort([4,3,2,1]))  # 2
print(min_swaps_to_sort([1,5,4,3,2]))# 3
```

---

## TL;DR / How these map to discrete math

* **Combinatorics**: `nCr`, Burnside’s Lemma (necklaces).
* **Number theory / modular arithmetic**: `nCr % p`, Fermat inverses.
* **Linear algebra**: matrix exponentiation for recurrences.
* **Permutation group theory**: cycle decomposition → min swaps.

---

If you want, I can:

* Solve 1–2 real contest problems end-to-end using these techniques, or
* Visualize cycle decomposition for a concrete example, or
* Print operation traces for the permutation-sorting algorithm.

Which would you like next?

Nice — here are **4 more classic competitive-programming problems** with the discrete-math idea, algorithm, complexity, and ready-to-run Python code + examples. I’ll keep each one focused and practical.

---

# 5) Sieve of Eratosthenes + Prime Factorization (using smallest prime factor)

**Problem:** Precompute primes up to `N` and answer prime-factorization queries fast.

**Math idea:** Sieve to find primes; store smallest prime factor (`spf`) for each `x` to factorize in `O(log x)`.

**Complexity:** Sieve `O(N log log N)`. Each factorization `O(number_of_prime_factors)` \~ `O(log x)`.

```python
def linear_sieve_spf(n):
    """Return list of primes up to n and spf array where spf[x] = smallest prime factor of x (>=2)"""
    spf = [0] * (n+1)
    primes = []
    for i in range(2, n+1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > n:
                break
            spf[i * p] = p
    return primes, spf

def factorize(x, spf):
    """Return prime factorization as list of (prime, exponent)"""
    res = []
    while x > 1:
        p = spf[x]
        cnt = 0
        while x % p == 0:
            x //= p
            cnt += 1
        res.append((p, cnt))
    return res

# Example
N = 100
primes, spf = linear_sieve_spf(N)
print("Primes up to 30:", [p for p in primes if p<=30])
print("Factorize 84:", factorize(84, spf))  # 84 = 2^2 * 3 * 7
```

---

# 6) Longest Increasing Subsequence (LIS) — patience sorting (O(n log n))

**Problem:** Given array `A`, find length (and optionally one LIS) of the longest strictly increasing subsequence.

**Math idea:** Maintain `tails[len] = smallest tail value for an increasing subsequence of length len+1`. Use binary search to place each element → `O(log n)` per element.

**Complexity:** `O(n log n)`.

```python
from bisect import bisect_left

def lis_length(A):
    tails = []
    for x in A:
        # for strictly increasing subsequence, use bisect_left on tails
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)

# Recover one LIS (with parent pointers)
def lis_sequence(A):
    n = len(A)
    tails = []
    idx = []          # idx[i] = index in A of tail for length i+1
    parent = [-1] * n
    for i, x in enumerate(A):
        pos = bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
            idx.append(i)
        else:
            tails[pos] = x
            idx[pos] = i
        if pos > 0:
            parent[i] = idx[pos-1]
    # reconstruct
    length = len(tails)
    lis = []
    cur = idx[-1] if idx else -1
    # find last index that corresponds to length-1
    for j in range(n-1, -1, -1):
        if parent[j] != -2 and (len(lis) == 0 and (tails and A[j] == tails[-1] or length==1) or False):
            pass
    # simpler reconstruction approach:
    # we track idx array as above; last index of LIS is idx[-1]
    cur = idx[-1] if idx else -1
    for _ in range(length):
        lis.append(A[cur])
        cur = parent[cur]
    lis.reverse()
    return length, lis

# Quick example (length only)
A = [3, 1, 2, 1, 8, 5, 6]
print("LIS length:", lis_length(A))  # 4 (e.g., 1,2,5,6)
```

> Note: the `lis_sequence` reconstruction above shows the usual parent/idx technique; for clarity I returned the length function only. If you want the full reconstruction code, I’ll send a shorter, robust version.

---

# 7) Minimum Spanning Tree (Kruskal’s algorithm) using DSU (Disjoint Set Union)

**Problem:** Given weighted undirected graph, compute MST total weight.

**Math idea:** Sort edges by weight and use DSU to greedily add smallest edge that connects two components.

**Complexity:** Sorting `O(E log E)`, DSU ops nearly O(α(N)) per op.

```python
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        a = self.find(a); b = self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1
        return True

def kruskal(n, edges):
    """
    n = number of vertices (0..n-1)
    edges = list of (w, u, v)
    returns (mst_weight, edges_used)
    """
    edges.sort()
    dsu = DSU(n)
    total = 0
    used = []
    for w, u, v in edges:
        if dsu.union(u, v):
            total += w
            used.append((u, v, w))
    return total, used

# Example
n = 4
edges = [
    (1, 0, 1),
    (4, 0, 2),
    (3, 0, 3),
    (2, 1, 2),
    (5, 2, 3)
]
print("MST weight, edges:", kruskal(n, edges))  # MST weight = 6 (1+2+3)
```

---

# 8) Chinese Remainder Theorem (CRT) — general pairwise (possibly non-coprime)

**Problem:** Solve system of congruences:

```
x ≡ a1 (mod m1)
x ≡ a2 (mod m2)
...
```

Return `(x, M)` where `M` is the lcm of moduli (or `None` if no solution).

**Math idea:** Combine congruences iteratively using extended Euclid to solve `x ≡ a1 (mod m1)` and `x ≡ a2 (mod m2)` into a single congruence, checking consistency when gcd>1.

**Complexity:** `O(k * log M)` where `k` = number of congruences.

```python
def extgcd(a, b):
    if b == 0: return (a, 1, 0)
    g, x1, y1 = extgcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def crt_pair(a1, m1, a2, m2):
    # solve x ≡ a1 (mod m1), x ≡ a2 (mod m2)
    # returns (x, lcm) or (None, None) if impossible
    g, p, q = extgcd(m1, m2)
    if (a2 - a1) % g != 0:
        return None, None
    # m1 * p + m2 * q = g
    lcm = m1 // g * m2
    # solution: x = a1 + m1 * ( (a2 - a1)/g * p mod (m2/g) )
    mult = (a2 - a1) // g
    mod = m2 // g
    x = (a1 + (m1 * ((mult * p) % mod))) % lcm
    return x, lcm

def crt(a, m):
    """a = list of remainders, m = list of moduli. returns (x, M) or (None, None)"""
    x, M = a[0], m[0]
    for ai, mi in zip(a[1:], m[1:]):
        x, M = crt_pair(x, M, ai, mi)
        if x is None:
            return None, None
    return x % M, M

# Example
a = [2, 3, 2]
m = [3, 5, 7]
print(crt(a, m))  # x ≡ 23 (mod 105) -> (23,105)

# Non-coprime example: x ≡ 2 (mod 6), x ≡ 8 (mod 10)
print(crt([2,8],[6,10]))  # solution exists? check -> (None, None) or value
```

---

## Quick recap of the 4 problems

* **Sieve + SPF:** prime precomputation and fast factorization — core for many number-theory tasks.
* **LIS (patience sorting):** `O(n log n)` to find longest increasing subsequence — used in sequence DP problems.
* **Kruskal + DSU:** build MST from edges — classic greedy using union-find.
* **CRT (general):** solve simultaneous congruences, including non-coprime moduli using extgcd.

---

Want me to:

* Expand **LIS reconstruction** into a short, correct implementation?
* Show a **live trace** (step-by-step) for one of these algorithms on a concrete input?
* Or pick 4 *different* problems (e.g., DP on trees, max-flow, FFT, or string algorithms)?


Let’s continue with **the next 4 classic problems**, each tied to **discrete math concepts** + competitive programming relevance.

---

# 9) Subset Sum / Knapsack (Dynamic Programming)

**Problem:** Given array `A` of size `n` and target `T`, check if any subset sums to `T`.

**Math idea:** This is based on **power set counting** (`2^n` subsets), but solved efficiently with **DP recurrence**:

$$
dp[i][s] = \text{True if we can form sum } s \text{ using first i elements}
$$

**Complexity:** `O(n*T)`.

```python
def subset_sum(A, T):
    n = len(A)
    dp = [False] * (T+1)
    dp[0] = True
    for x in A:
        for s in range(T, x-1, -1):
            dp[s] = dp[s] or dp[s-x]
    return dp[T]

# Example
A = [3, 34, 4, 12, 5, 2]
T = 9
print("Subset sum exists:", subset_sum(A, T))  # True (4+5)
```

---

# 10) Maximum Bipartite Matching (Graph Theory + Hall’s Theorem)

**Problem:** Assign workers to jobs if worker i can do job j. Find maximum matching.

**Math idea:** A **bipartite graph** problem solved with **DFS augmenting paths** (Hungarian algorithm or Hopcroft–Karp).

**Complexity:** Hopcroft-Karp → `O(E √V)`. Here: simpler DFS augmenting → `O(VE)`.

```python
def bpm(u, matchR, seen, bpGraph):
    for v in range(len(bpGraph[0])):
        if bpGraph[u][v] and not seen[v]:
            seen[v] = True
            if matchR[v] == -1 or bpm(matchR[v], matchR, seen, bpGraph):
                matchR[v] = u
                return True
    return False

def maxBPM(bpGraph):
    m = len(bpGraph)      # workers
    n = len(bpGraph[0])   # jobs
    matchR = [-1] * n
    result = 0
    for i in range(m):
        seen = [False] * n
        if bpm(i, matchR, seen, bpGraph):
            result += 1
    return result

# Example
bpGraph = [
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
]
print("Max bipartite matching:", maxBPM(bpGraph))  # 3
```

---

# 11) Fast Fourier Transform (FFT) for Polynomial Multiplication

**Problem:** Multiply two polynomials (or big integers) efficiently.

**Math idea:** Discrete Fourier Transform (DFT) over complex roots of unity. FFT reduces from `O(n^2)` to `O(n log n)`.

**Complexity:** `O(n log n)`.

```python
import cmath

def fft(a, invert=False):
    n = len(a)
    if n == 1:
        return a
    even = fft(a[0::2], invert)
    odd  = fft(a[1::2], invert)
    ang = 2*cmath.pi/n * (-1 if invert else 1)
    w, wn = 1, cmath.exp(ang*1j)
    y = [0]*n
    for k in range(n//2):
        y[k]       = even[k] + w * odd[k]
        y[k+n//2]  = even[k] - w * odd[k]
        w *= wn
    if invert:
        for i in range(n):
            y[i] /= 2
    return y

def multiply_poly(a, b):
    n = 1
    while n < len(a)+len(b):
        n <<= 1
    fa = fft(a+[0]*(n-len(a)))
    fb = fft(b+[0]*(n-len(b)))
    for i in range(n):
        fa[i] *= fb[i]
    res = fft(fa, invert=True)
    return [round(x.real) for x in res]

# Example: (1 + 2x + 3x^2) * (4 + 5x)
A = [1,2,3]
B = [4,5]
print("Polynomial multiplication:", multiply_poly(A,B))  # [4,13,22,15]
```

---

# 12) Matrix Exponentiation (Linear Recurrence, Fibonacci)

**Problem:** Compute nth Fibonacci number modulo M in `O(log n)`.

**Math idea:** Fibonacci recurrence as matrix:

$$
\begin{bmatrix} F_{n+1} \\ F_n \end{bmatrix} =
\begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}
\cdot
\begin{bmatrix} F_n \\ F_{n-1} \end{bmatrix}
$$

**Complexity:** `O(log n)`.

```python
def matmul(A, B, mod):
    return [[(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
             (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
            [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
             (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]]

def matpow(A, n, mod):
    res = [[1,0],[0,1]]
    while n > 0:
        if n & 1:
            res = matmul(res, A, mod)
        A = matmul(A, A, mod)
        n >>= 1
    return res

def fib(n, mod=10**9+7):
    if n==0: return 0
    F = [[1,1],[1,0]]
    M = matpow(F, n-1, mod)
    return M[0][0]

# Example
print("Fib(10) mod 1e9+7:", fib(10))  # 55
```

---

## ✅ Recap of Next 4

* **Subset Sum / Knapsack** → combinatorics + DP recurrence.
* **Bipartite Matching** → graph theory + Hall’s theorem.
* **FFT** → discrete transforms + polynomial/integer multiplication.
* **Matrix Exponentiation** → linear recurrences via algebra.

---

⚡ Question for you:
Do you want me to keep going with **more number theory + graph + string algorithms**, or should I also cover **probability/statistics-based CP problems** (like expected values, random walks, DP on probabilities)?


Perfect 👍 Let’s move to the **next 4 classic competitive programming problems**, each mapped to a discrete math concept.

---

# 13) Euler’s Totient Function (φ) Precomputation

**Problem:** For all numbers up to `N`, compute φ(n) (count of integers ≤ n coprime to n).

**Math idea:**

$$
\varphi(n) = n \prod_{p|n} \left(1 - \frac{1}{p}\right)
$$

Use a sieve-like method to compute efficiently.

**Complexity:** `O(N log log N)`.

```python
def compute_totients(n):
    phi = list(range(n+1))
    for i in range(2, n+1):
        if phi[i] == i:  # prime
            for j in range(i, n+1, i):
                phi[j] -= phi[j] // i
    return phi

# Example
phi = compute_totients(20)
print("Totients up to 20:", phi[1:21])
# φ(9)=6, φ(10)=4, φ(12)=4
```

---

# 14) Strongly Connected Components (Kosaraju’s Algorithm)

**Problem:** Find SCCs in a directed graph.

**Math idea:** SCC = equivalence relation under “mutual reachability”.
Kosaraju: DFS order + reverse graph DFS.

**Complexity:** `O(V+E)`.

```python
from collections import defaultdict

def kosaraju(n, edges):
    g = defaultdict(list)
    gr = defaultdict(list)
    for u,v in edges:
        g[u].append(v)
        gr[v].append(u)

    visited = [False]*n
    order = []
    def dfs1(u):
        visited[u] = True
        for v in g[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    comp = []
    visited = [False]*n
    def dfs2(u, cur):
        visited[u] = True
        cur.append(u)
        for v in gr[u]:
            if not visited[v]:
                dfs2(v, cur)

    for u in reversed(order):
        if not visited[u]:
            cur = []
            dfs2(u, cur)
            comp.append(cur)
    return comp

# Example
edges = [(0,1),(1,2),(2,0),(1,3),(3,4)]
print("SCCs:", kosaraju(5, edges))  
# Output: [[0,2,1],[3],[4]]
```

---

# 15) Maximum Flow (Ford-Fulkerson / Edmonds-Karp)

**Problem:** Given capacity graph, compute max flow from `s` to `t`.

**Math idea:** Flow networks are classic in discrete math optimization.
Use BFS to find augmenting paths (Edmonds-Karp).

**Complexity:** `O(V * E^2)` (ok for medium constraints).

```python
from collections import deque

def bfs(C, F, s, t, parent):
    n = len(C)
    visited = [False]*n
    q = deque([s])
    visited[s] = True
    while q:
        u = q.popleft()
        for v in range(n):
            if not visited[v] and C[u][v]-F[u][v] > 0:
                parent[v] = u
                visited[v] = True
                q.append(v)
                if v == t:
                    return True
    return False

def maxflow(C, s, t):
    n = len(C)
    F = [[0]*n for _ in range(n)]
    parent = [-1]*n
    flow = 0
    while bfs(C,F,s,t,parent):
        path_flow = float('inf')
        v = t
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, C[u][v]-F[u][v])
            v = parent[v]
        v = t
        while v != s:
            u = parent[v]
            F[u][v] += path_flow
            F[v][u] -= path_flow
            v = parent[v]
        flow += path_flow
    return flow

# Example
C = [[0,16,13,0,0,0],
     [0,0,10,12,0,0],
     [0,4,0,0,14,0],
     [0,0,9,0,0,20],
     [0,0,0,7,0,4],
     [0,0,0,0,0,0]]
print("Max flow:", maxflow(C, 0, 5))  # 23
```

---

# 16) KMP String Matching (Knuth-Morris-Pratt)

**Problem:** Find all occurrences of pattern `P` in text `T`.

**Math idea:** Uses **prefix function (π)** from automata theory (discrete structures).

**Complexity:** `O(n+m)`.

```python
def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    lps = [0]*m
    j = 0
    for i in range(1, m):
        while j>0 and pattern[i]!=pattern[j]:
            j = lps[j-1]
        if pattern[i]==pattern[j]:
            j+=1
            lps[i]=j

    res = []
    j = 0
    for i in range(n):
        while j>0 and text[i]!=pattern[j]:
            j = lps[j-1]
        if text[i]==pattern[j]:
            j+=1
        if j==m:
            res.append(i-m+1)
            j = lps[j-1]
    return res

# Example
text = "ababcabcabababd"
pattern = "ababd"
print("Pattern found at:", kmp_search(text, pattern))  # [10]
```

---

## ✅ Recap of this batch

* **Euler Totient (φ sieve)** → number theory + multiplicative functions.
* **SCC (Kosaraju)** → graph theory + equivalence classes.
* **Max Flow** → network flow optimization.
* **KMP** → automata & string matching.

---

👉 Want me to push into the **next set of 4** covering:

* Probability/Expected Value problems (Dice, Random Walks)
* Game theory (Grundy numbers / Nim)
* Data structures + discrete math (Fenwick tree / segment tree)
* Geometry + discrete math (Convex Hull / Graham scan)?



🔥 Awesome, now let’s tackle **4 more problem types** that bring in **probability, game theory, data structures, and computational geometry** — all rooted in discrete mathematics.

---

# 17) Expected Value in Dice Rolls 🎲

**Problem:** Roll a fair 6-sided die `n` times. Find expected sum.

**Math idea:** Linearity of expectation:

$$
E[\text{sum}] = n \cdot E[\text{single roll}] = n \cdot 3.5
$$

This generalizes to many expected-value problems in CP.

**Complexity:** `O(1)`.

```python
def expected_dice_sum(n):
    return n * 3.5

# Example
print("Expected sum of 10 dice:", expected_dice_sum(10))  # 35.0
```

✅ In CP, this extends to: *expected number of coin tosses to get heads*, *expected steps in random walks*, etc.

---

# 18) Game Theory (Nim Game, Grundy Numbers) ♟️

**Problem:** Two players alternately take stones from piles. Each turn, remove any number from one pile. Who wins?

**Math idea:** Sprague-Grundy theorem. A position is losing if **xor of pile sizes = 0**, otherwise winning.

**Complexity:** `O(n)` for checking outcome.

```python
def nim_game(piles):
    xor_sum = 0
    for p in piles:
        xor_sum ^= p
    return "First" if xor_sum != 0 else "Second"

# Example
print("Winner:", nim_game([3,4,5]))  # First wins
print("Winner:", nim_game([1,1]))    # Second wins
```

✅ In CP, Grundy numbers extend to arbitrary impartial games (e.g., splitting piles, graph games).

---

# 19) Fenwick Tree (Binary Indexed Tree) 📊

**Problem:** Support prefix sums & point updates in `O(log n)`.

**Math idea:** Uses binary representation and properties of discrete logarithms (least significant bit).

**Complexity:** Update/query `O(log n)`.

```python
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.bit = [0]*(n+1)
    def update(self, i, delta):
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i
    def query(self, i):
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s
    def range_query(self, l, r):
        return self.query(r) - self.query(l-1)

# Example
ft = FenwickTree(10)
ft.update(3, 5)  # A[3]+=5
ft.update(7, 2)
print("Prefix sum [1..7]:", ft.query(7))   # 7
print("Range sum [3..7]:", ft.range_query(3,7))  # 7
```

✅ Used in inversion counting, frequency tables, order statistics.

---

# 20) Convex Hull (Computational Geometry, Graham Scan) 🟦

**Problem:** Given points, find convex hull (smallest convex polygon containing all points).

**Math idea:** Sort points + use **cross product** to detect left turns.

**Complexity:** `O(n log n)`.

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

# Example
pts = [(0,0),(1,1),(2,2),(2,0),(2,4),(3,3),(0,3)]
print("Convex Hull:", convex_hull(pts))  
# [(0,0),(2,0),(3,3),(2,4),(0,3)]
```

✅ Used in closest pair problems, minimum perimeter polygon, computational geometry contests.

---

# ✅ Recap of this batch

* **Expected Value (Dice Rolls):** probability + linearity of expectation.
* **Nim Game (Grundy):** game theory + xor property.
* **Fenwick Tree:** data structures + binary decomposition.
* **Convex Hull (Graham Scan):** geometry + orientation tests.

---

👉 Want me to continue with the **next 4 covering string algorithms (Suffix Array/Trie), matrix-based graph algorithms (Floyd–Warshall), probabilistic methods (Monte Carlo), and DP on subsets (bitmask DP)?**


Perfect 🚀 Let’s continue with **the next 4 problems**, covering **strings, graphs, probability, and bitmask DP** — all rooted in discrete mathematics.

---

# 21) **Suffix Array & LCP (String Algorithms)** 📜

**Problem:** Given a string `s`, find its suffixes in lexicographic order.

**Math idea:** Sorting all suffixes + longest common prefix (LCP) uses discrete ordering relations.

**Complexity:** `O(n log n)` with sorting.

```python
def suffix_array(s):
    return sorted(range(len(s)), key=lambda i: s[i:])

def build_lcp(s, sa):
    n = len(s)
    rank = [0]*n
    for i, si in enumerate(sa):
        rank[si] = i
    lcp, k = [0]*(n-1), 0
    for i in range(n):
        if rank[i] == n-1: 
            k = 0
            continue
        j = sa[rank[i]+1]
        while i+k<n and j+k<n and s[i+k]==s[j+k]:
            k += 1
        lcp[rank[i]] = k
        if k: k -= 1
    return lcp

# Example
s = "banana"
sa = suffix_array(s)
lcp = build_lcp(s, sa)
print("Suffix Array:", sa)
print("LCP:", lcp)
```

✅ Used in substring search, distinct substring counting, suffix automaton problems.

---

# 22) **Floyd–Warshall Algorithm (Graph, Matrix-based)** 🌐

**Problem:** Find all-pairs shortest paths in weighted graph.

**Math idea:** Dynamic programming over path lengths, matrix recurrence:

$$
d[i][j] = \min(d[i][j],\ d[i][k] + d[k][j])
$$

**Complexity:** `O(n^3)`.

```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [row[:] for row in graph]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

INF = 10**9
graph = [
    [0, 3, INF, 7],
    [8, 0, 2, INF],
    [5, INF, 0, 1],
    [2, INF, INF, 0]
]

dist = floyd_warshall(graph)
print("All pairs shortest paths:")
for row in dist: print(row)
```

✅ Used in dense graphs, transitive closure, minimum cycle detection.

---

# 23) **Monte Carlo Simulation (Probabilistic Method)** 🎲

**Problem:** Estimate value of π by random sampling inside a square.

**Math idea:** Probability & geometric probability.

$$
\pi \approx 4 \cdot \frac{\text{points inside circle}}{\text{total points}}
$$

**Complexity:** `O(N)` trials.

```python
import random

def estimate_pi(trials=100000):
    inside = 0
    for _ in range(trials):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / trials

print("Estimated π:", estimate_pi(100000))
```

✅ Used in approximation algorithms, randomized primality testing, expected values.

---

# 24) **Bitmask DP (Traveling Salesman Problem)** 🗺️

**Problem:** Minimum Hamiltonian path visiting all nodes (TSP).

**Math idea:** Subset DP — state represented by bitmask.

$$
dp[mask][i] = \min(dp[mask][i], dp[mask \setminus \{i\}][j] + cost[j][i])
$$

**Complexity:** `O(n^2 2^n)`.

```python
from functools import lru_cache

def tsp(cost):
    n = len(cost)
    @lru_cache(None)
    def dp(mask, i):
        if mask == (1<<n)-1:
            return cost[i][0]  # return to start
        ans = float('inf')
        for j in range(n):
            if not (mask & (1<<j)):
                ans = min(ans, cost[i][j] + dp(mask | (1<<j), j))
        return ans
    return dp(1, 0)

cost = [
    [0, 20, 42, 35],
    [20, 0, 30, 34],
    [42, 30, 0, 12],
    [35, 34, 12, 0]
]
print("TSP minimum cost:", tsp(cost))
```

✅ Used in TSP, subset selection, set cover, bitmask DP problems in CP.

---

# ✅ Recap of this batch

* **Suffix Array & LCP:** string algorithms & order theory.
* **Floyd–Warshall:** matrix-based DP for shortest paths.
* **Monte Carlo Simulation:** probability & approximation.
* **Bitmask DP (TSP):** exponential DP over subsets.

---

👉 Do you want me to **wrap up with the final 4 problems** (covering **linear programming, number theory with CRT, SAT/logic problems, and combinatorial optimization like max flow**) so we complete the full tour of 28 problems?


🔥 Awesome — let’s finish strong with the **last 4 problems** (25–28) that show how **discrete mathematics powers optimization, number theory, logic, and graph algorithms**.

---

# 25) **Linear Programming (Optimization)** 📈

**Problem:** Maximize profit given resource constraints (e.g., knapsack-like).

**Math idea:** Discrete → Continuous relaxation → LP.

* Many integer programming problems in CP can be approximated/solved with LP.
* Solve using **Simplex** or specialized solvers.

**Complexity:** Polynomial time (with interior-point).

```python
from scipy.optimize import linprog

# Maximize 3x + 2y subject to:
# 2x + y <= 20
# x + 2y <= 20
# x,y >= 0

c = [-3, -2]  # maximize => minimize -profit
A = [[2, 1], [1, 2]]
b = [20, 20]

res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
print("Optimal profit:", -res.fun)
print("x,y:", res.x)
```

✅ LP is used in scheduling, flows, matching, optimization problems.

---

# 26) **Chinese Remainder Theorem (CRT, Number Theory)** 🔢

**Problem:** Solve system of congruences.

$$
x \equiv 2 \ (\text{mod } 3), \quad
x \equiv 3 \ (\text{mod } 5), \quad
x \equiv 2 \ (\text{mod } 7)
$$

**Math idea:** Constructive theorem — unique solution modulo product of moduli (if coprime).

**Complexity:** `O(n log M)` with modular inverses.

```python
def chinese_remainder(n, a):
    sum = 0
    prod = 1
    for ni in n:
        prod *= ni
    for ni, ai in zip(n, a):
        p = prod // ni
        inv = pow(p, -1, ni)  # modular inverse
        sum += ai * inv * p
    return sum % prod

n = [3, 5, 7]
a = [2, 3, 2]
print("CRT Solution:", chinese_remainder(n, a))  # 23
```

✅ Appears in cryptography (RSA), modular arithmetic CP problems.

---

# 27) **Boolean Satisfiability (SAT, Logic)** 🧩

**Problem:** Given Boolean formula in CNF, decide satisfiability.

**Math idea:** Discrete logic, truth tables, 2-SAT solvable in polynomial time with graph SCCs.

**Complexity:** NP-complete (general SAT).

👉 Example: **2-SAT using implication graph**.

```python
from collections import defaultdict

def two_sat(n, clauses):
    graph = defaultdict(list)
    def add_implication(x, y):
        graph[x].append(y)

    for a, b in clauses:
        add_implication(-a, b)  # ¬a ⇒ b
        add_implication(-b, a)  # ¬b ⇒ a

    # Kosaraju for SCC
    order, visited = [], set()
    def dfs(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
        order.append(u)

    for u in list(graph):
        if u not in visited: dfs(u)

    comp, visited = {}, set()
    def dfs_rev(u, label):
        comp[u] = label
        visited.add(u)
        for v in [x for x,y in graph.items() if u in y]:
            if v not in visited:
                dfs_rev(v, label)

    for u in reversed(order):
        if u not in visited:
            dfs_rev(u, u)

    for i in range(1, n+1):
        if comp.get(i) == comp.get(-i):
            return False
    return True

# Example: (x1 ∨ x2) ∧ (¬x1 ∨ x2) ∧ (¬x2 ∨ x1)
print("Satisfiable:", two_sat(2, [(1,2), (-1,2), (-2,1)]))
```

✅ SAT/2-SAT is used in constraint satisfaction, scheduling, puzzles.

---

# 28) **Maximum Flow (Combinatorial Optimization)** 🌊

**Problem:** Given a network with capacities, find max flow from source to sink.

**Math idea:** Discrete optimization, augmenting paths, min-cut theorem.

**Complexity:**

* Edmonds-Karp: `O(VE^2)`
* Dinic: `O(E√V)`

```python
from collections import deque

def bfs(capacity, source, sink, parent):
    visited = set([source])
    q = deque([source])
    while q:
        u = q.popleft()
        for v in range(len(capacity)):
            if v not in visited and capacity[u][v] > 0:
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return True
                q.append(v)
    return False

def max_flow(capacity, source, sink):
    n = len(capacity)
    parent = [-1]*n
    flow = 0
    while bfs(capacity, source, sink, parent):
        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, capacity[u][v])
            v = parent[v]
        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = parent[v]
        flow += path_flow
    return flow

capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
print("Max Flow:", max_flow([row[:] for row in capacity], 0, 5))  # 23
```

✅ Applications: bipartite matching, circulation, scheduling, network routing.

---

# 🎯 Recap of Final 4

* **Linear Programming:** optimization with constraints.
* **Chinese Remainder Theorem:** modular arithmetic & number theory.
* **SAT/2-SAT:** logic & satisfiability.
* **Max Flow:** combinatorial optimization on graphs.

---

✨ That completes our **28-problem walkthrough** of how **discrete mathematics underpins competitive programming**.

👉 Do you want me to now create a **mind map / visual roadmap** connecting all 28 problems to the branches of discrete math (sets, logic, combinatorics, algebra, number theory, probability, graph theory, optimization)?

Perfect! Let’s create a **visual roadmap / mind map** connecting all **28 problems** to the **branches of discrete mathematics**. I’ll structure it as a **text-based mind map** you could later convert to a diagram if needed.

---

# **Discrete Mathematics → Competitive Programming Problems Mind Map**

```
Discrete Mathematics
├── 1. Number Theory
│   ├── 1) Sieve of Eratosthenes + SPF (Prime factorization)
│   ├── 13) Euler's Totient Function φ
│   ├── 26) Chinese Remainder Theorem (CRT)
│   └── Modular arithmetic problems in CP
│
├── 2. Combinatorics
│   ├── 5) Subset Sum / Knapsack (DP)
│   ├── 24) Bitmask DP (TSP)
│   └── Counting / permutation / combination problems
│
├── 3. Graph Theory
│   ├── 6) Longest Increasing Subsequence (sequence DP as DAG)
│   ├── 7) Minimum Spanning Tree (Kruskal + DSU)
│   ├── 14) Strongly Connected Components (Kosaraju)
│   ├── 22) Floyd–Warshall (All-pairs shortest paths)
│   ├── 18) Game Theory (Nim, Grundy numbers → graph view)
│   └── 28) Maximum Flow (Edmonds-Karp / Dinic)
│
├── 4. Probability & Expectation
│   ├── 17) Expected Value (Dice rolls)
│   └── 23) Monte Carlo Simulation
│
├── 5. Algebra & Linear Recurrences
│   ├── 8) Matrix Exponentiation (Fibonacci / Linear Recurrence)
│   └── Linear system solving (LP relaxation)
│
├── 6. Discrete Geometry
│   └── 20) Convex Hull (Graham Scan)
│
├── 7. String Algorithms & Automata
│   ├── 21) Suffix Array & LCP
│   └── 20) KMP String Matching
│
├── 8. Optimization / Linear Programming
│   ├── 25) Linear Programming (Simplex)
│   └── Max Flow & Min Cut (28)
│
├── 9. Logic / Boolean Algebra
│   └── 27) Boolean Satisfiability (SAT / 2-SAT)
│
├── 10. Data Structures & Discrete Math
│   └── 19) Fenwick Tree (Binary Indexed Tree)
│
├── 11. FFT / Polynomials
│   └── 11) Fast Fourier Transform (Polynomial multiplication)
│
└── 12. Others / Advanced
    └── Sequence problems (e.g., LIS, DP over subsets, linear recurrences)
```

---

### ✅ Notes on the Mind Map

* **Each branch** maps a **discrete math concept** to **CP problems**.
* Many CP problems **overlap multiple branches**: e.g.,

  * MST uses Graph + DSU (Algebraic structure).
  * Bitmask DP (Combinatorics + Subset representation).
  * Monte Carlo (Probability + Approximation).
* This roadmap can be **expanded into a visual diagram** with nodes for problems and links to the math branch.

---

If you want, I can **draw a proper visual mind map diagram** for all 28 problems **with colored branches and arrows**, ready to use for study or teaching.

Do you want me to do that?



