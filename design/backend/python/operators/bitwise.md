Sure! Python provides a set of **bitwise operators** that operate on integers at the **bit level**. These are useful in performance-critical applications, low-level programming, cryptography, image processing, etc.

---

## 🔹 List of Bitwise Operators in Python

| Operator    | Symbol | Meaning                    |            |
| ----------- | ------ | -------------------------- | ---------- |
| AND         | `&`    | Bitwise AND                |            |
| OR          | \`     | \`                         | Bitwise OR |
| XOR         | `^`    | Bitwise XOR (exclusive OR) |            |
| NOT         | `~`    | Bitwise NOT (inversion)    |            |
| Left Shift  | `<<`   | Shift bits left            |            |
| Right Shift | `>>`   | Shift bits right           |            |

---

## 🔸 1. Bitwise AND (`&`)

Sets each bit to 1 **if both bits are 1**.

```python
a = 5        # 0101
b = 3        # 0011
print(a & b) # 0001 => 1
```

---

## 🔸 2. Bitwise OR (`|`)

Sets each bit to 1 **if at least one bit is 1**.

```python
a = 5        # 0101
b = 3        # 0011
print(a | b) # 0111 => 7
```

---

## 🔸 3. Bitwise XOR (`^`)

Sets each bit to 1 **only if bits are different**.

```python
a = 5        # 0101
b = 3        # 0011
print(a ^ b) # 0110 => 6
```

---

## 🔸 4. Bitwise NOT (`~`)

Inverts all bits (i.e., `~x` is `-(x + 1)` in two’s complement).

```python
a = 5         # 0000 0101
print(~a)     # 1111 1010 => -6
```

🧠 Explanation:

```
~5 = -(5 + 1) = -6
```

---

## 🔸 5. Left Shift (`<<`)

Shifts bits to the left (adds zeros on the right).

```python
a = 5         # 0000 0101
print(a << 1) # 0000 1010 => 10
print(a << 2) # 0001 0100 => 20
```

🧠 Left shift by `n` is same as multiplying by `2ⁿ`.

---

## 🔸 6. Right Shift (`>>`)

Shifts bits to the right (drops bits from the right).

```python
a = 5         # 0000 0101
print(a >> 1) # 0000 0010 => 2
print(a >> 2) # 0000 0001 => 1
```

🧠 Right shift by `n` is same as integer division by `2ⁿ`.

---

## 🧪 Bonus: Binary Representation

Use `bin(x)` to view binary:

```python
print(bin(5))  # '0b101'
print(bin(~5)) # '-0b110'
```

---

## ✅ Summary Table

| Expression | Result | Binary                            |        |               |
| ---------- | ------ | --------------------------------- | ------ | ------------- |
| `5 & 3`    | 1      | `0101 & 0011 = 0001`              |        |               |
| \`5        | 3\`    | 7                                 | \`0101 | 0011 = 0111\` |
| `5 ^ 3`    | 6      | `0101 ^ 0011 = 0110`              |        |               |
| `~5`       | -6     | `~0101 = 1010` (two's complement) |        |               |
| `5 << 1`   | 10     | `0101 → 1010`                     |        |               |
| `5 >> 1`   | 2      | `0101 → 0010`                     |        |               |

---

Let me know if you’d like real-world applications or advanced tricks using bitwise ops (e.g., checking even/odd, swapping values without temp variable, bit masks, etc.)!
