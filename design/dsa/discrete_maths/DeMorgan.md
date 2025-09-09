

---

## 🧠 **De Morgan’s Laws**

De Morgan’s Laws are **fundamental rules of logic and set theory** that describe how **negations** interact with **AND (∧)** and **OR (∨)** operations.

---

### 🔹 In **Propositional Logic**

For any two logical statements $A$ and $B$:

1.

$$
\neg(A \land B) \equiv (\neg A) \lor (\neg B)
$$

> "NOT (A AND B)" is equivalent to "(NOT A) OR (NOT B)".

2.

$$
\neg(A \lor B) \equiv (\neg A) \land (\neg B)
$$

> "NOT (A OR B)" is equivalent to "(NOT A) AND (NOT B)".

---

### 🔹 In **Set Theory**

For any two sets $A$ and $B$:

1.

$$
(A \cap B)^c = A^c \cup B^c
$$

> The complement of an intersection = union of complements.

2.

$$
(A \cup B)^c = A^c \cap B^c
$$

> The complement of a union = intersection of complements.

---

### 🔹 Truth Table Verification

| A | B | $A \land B$ | $\neg(A \land B)$ | $\neg A \lor \neg B$ |
| - | - | ----------- | ----------------- | -------------------- |
| T | T | T           | F                 | F                    |
| T | F | F           | T                 | T                    |
| F | T | F           | T                 | T                    |
| F | F | F           | T                 | T                    |

✅ Columns match → first law holds.
The second law can be verified the same way.

---

### 🔹 Everyday Example

* Statement: "It is **not** true that I will eat pizza **and** drink soda."
  Equivalent: "Either I will **not** eat pizza, **or** I will **not** drink soda."

* Statement: "It is **not** true that I will eat pizza **or** drink soda."
  Equivalent: "I will **not** eat pizza **and** I will **not** drink soda."

---

✅ So De Morgan’s Laws basically show how **negation flips AND ↔ OR** (and ∩ ↔ ∪ in sets).

---

Do you want me to also draw a **Venn diagram** to visually show how the set version works?
