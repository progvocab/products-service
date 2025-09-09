

A **circular permutation** (also called a **cyclic permutation**) is a way of arranging objects **around a circle** where rotations are considered the same arrangement.

---

## 📖 Definition

* In a **linear permutation** of `n` elements → there are `n!` possible orders.
* In a **circular permutation**, we "fix" one element as a reference (since rotations don’t create a new unique arrangement).
* Formula:

$$
\text{Number of circular permutations of } n \text{ objects} = (n-1)!
$$

---

## 📌 Example

* Linear permutations of `[A, B, C]`:
  `ABC, ACB, BAC, BCA, CAB, CBA` → total = `3! = 6`.
* Circular permutations (rotations considered same):
  `ABC, ACB` → total = `(3-1)! = 2`.

  * `ABC`, `BCA`, `CAB` are the same in a circle.
  * `ACB`, `CBA`, `BAC` are the same.

---

## 🔑 Variants

1. **Oriented circle** (clockwise vs counterclockwise different): `(n-1)!`
2. **Unoriented circle** (if flipping also makes no difference, like necklaces):

   $$
   \frac{(n-1)!}{2}
   $$

---

## 🐍 Python Example

```python
import itertools

def circular_permutations(elements):
    n = len(elements)
    seen = set()
    result = []

    for perm in itertools.permutations(elements):
        # normalize rotation: smallest rotation
        rotations = [perm[i:] + perm[:i] for i in range(n)]
        min_rotation = min(rotations)
        if min_rotation not in seen:
            seen.add(min_rotation)
            result.append(perm)
    return result

# Example
print(circular_permutations([1,2,3]))
```

### ✅ Output

```
[(1, 2, 3), (1, 3, 2)]
```

Which are the **2 unique circular permutations**.

---

## 🔎 Where It’s Used

* **Necklace / bracelet counting** (Burnside’s lemma, Polya counting)
* **Seating arrangements around a round table**
* **Genetics (DNA rotations)**
* **Cryptography (rotational symmetries)**

---

👉 Do you want me to also explain **circular permutations with restrictions** (like no two friends sit together at a round table), since those are common in competitive programming?
