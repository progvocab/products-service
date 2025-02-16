# **Python Operators with Examples**  

Operators in Python are special symbols that perform operations on variables and values. Python provides various types of operators:  

### **1️⃣ Arithmetic Operators**
Used for mathematical operations.  

| Operator | Description | Example |
|----------|------------|---------|
| `+` | Addition | `5 + 3  # 8` |
| `-` | Subtraction | `5 - 3  # 2` |
| `*` | Multiplication | `5 * 3  # 15` |
| `/` | Division | `5 / 2  # 2.5` |
| `//` | Floor Division | `5 // 2  # 2` |
| `%` | Modulus | `5 % 2  # 1` |
| `**` | Exponentiation | `5 ** 2  # 25` |

#### **Example**
```python
a, b = 10, 3
print(a + b)  # 13
print(a - b)  # 7
print(a * b)  # 30
print(a / b)  # 3.333
print(a // b) # 3
print(a % b)  # 1
print(a ** b) # 1000
```

---

### **2️⃣ Comparison (Relational) Operators**
Used for comparing values and return `True` or `False`.

| Operator | Description | Example |
|----------|------------|---------|
| `==` | Equal to | `5 == 5  # True` |
| `!=` | Not equal to | `5 != 3  # True` |
| `>` | Greater than | `5 > 3  # True` |
| `<` | Less than | `5 < 3  # False` |
| `>=` | Greater than or equal to | `5 >= 5  # True` |
| `<=` | Less than or equal to | `3 <= 5  # True` |

#### **Example**
```python
x, y = 10, 20
print(x == y)  # False
print(x != y)  # True
print(x > y)   # False
print(x < y)   # True
print(x >= y)  # False
print(x <= y)  # True
```

---

### **3️⃣ Logical Operators**
Used for combining conditional statements.

| Operator | Description | Example |
|----------|------------|---------|
| `and` | Returns `True` if both conditions are `True` | `(5 > 3) and (10 > 5)  # True` |
| `or` | Returns `True` if at least one condition is `True` | `(5 > 3) or (10 < 5)  # True` |
| `not` | Reverses the result | `not(5 > 3)  # False` |

#### **Example**
```python
a, b = True, False
print(a and b)  # False
print(a or b)   # True
print(not a)    # False
```

---

### **4️⃣ Bitwise Operators**
Used for bit-level operations.

| Operator | Description | Example |
|----------|------------|---------|
| `&` | Bitwise AND | `5 & 3  # 1` |
| `|` | Bitwise OR | `5 | 3  # 7` |
| `^` | Bitwise XOR | `5 ^ 3  # 6` |
| `~` | Bitwise NOT | `~5  # -6` |
| `<<` | Left shift | `5 << 1  # 10` |
| `>>` | Right shift | `5 >> 1  # 2` |

#### **Example**
```python
x, y = 5, 3
print(x & y)  # 1
print(x | y)  # 7
print(x ^ y)  # 6
print(~x)     # -6
print(x << 1) # 10
print(x >> 1) # 2
```

---

### **5️⃣ Assignment Operators**
Used for assigning values to variables.

| Operator | Description | Example |
|----------|------------|---------|
| `=` | Assign value | `x = 5` |
| `+=` | Add & assign | `x += 3  # x = x + 3` |
| `-=` | Subtract & assign | `x -= 3  # x = x - 3` |
| `*=` | Multiply & assign | `x *= 3  # x = x * 3` |
| `/=` | Divide & assign | `x /= 3  # x = x / 3` |
| `//=` | Floor divide & assign | `x //= 3  # x = x // 3` |
| `%=` | Modulus & assign | `x %= 3  # x = x % 3` |
| `**=` | Exponentiate & assign | `x **= 3  # x = x ** 3` |
| `&=` | Bitwise AND & assign | `x &= 3  # x = x & 3` |
| `|=` | Bitwise OR & assign | `x |= 3  # x = x | 3` |
| `^=` | Bitwise XOR & assign | `x ^= 3  # x = x ^ 3` |
| `<<=` | Left shift & assign | `x <<= 3  # x = x << 3` |
| `>>=` | Right shift & assign | `x >>= 3  # x = x >> 3` |

#### **Example**
```python
x = 5
x += 3
print(x)  # 8
```

---

### **6️⃣ Membership Operators**
Used to check membership in sequences (lists, tuples, strings, etc.).

| Operator | Description | Example |
|----------|------------|---------|
| `in` | Returns `True` if value exists | `'a' in 'apple'  # True` |
| `not in` | Returns `True` if value does not exist | `'z' not in 'apple'  # True` |

#### **Example**
```python
fruits = ['apple', 'banana', 'cherry']
print('apple' in fruits)  # True
print('grape' not in fruits)  # True
```

---

### **7️⃣ Identity Operators**
Used to compare memory locations of objects.

| Operator | Description | Example |
|----------|------------|---------|
| `is` | Returns `True` if same object | `x is y` |
| `is not` | Returns `True` if different objects | `x is not y` |

#### **Example**
```python
a = [1, 2, 3]
b = a
c = [1, 2, 3]

print(a is b)   # True (same object)
print(a is c)   # False (different objects)
print(a == c)   # True (same values)
```

---

### **8️⃣ Ternary Operator**
Short-hand for if-else.

#### **Example**
```python
x, y = 10, 20
min_value = x if x < y else y
print(min_value)  # 10
```

---

### **9️⃣ Operator Precedence**
Operator precedence determines the order of execution.

#### **Precedence Table (Highest to Lowest)**
| Precedence | Operators |
|------------|----------|
| 1 (Highest) | `()`, `[]`, `{}` (Parentheses, indexing) |
| 2 | `**` (Exponentiation) |
| 3 | `+x, -x, ~x` (Unary plus, minus, bitwise NOT) |
| 4 | `*, /, //, %` (Multiplication, division, floor division, modulus) |
| 5 | `+, -` (Addition, subtraction) |
| 6 | `<<, >>` (Bitwise shifts) |
| 7 | `&` (Bitwise AND) |
| 8 | `^` (Bitwise XOR) |
| 9 | `|` (Bitwise OR) |
| 10 | `==, !=, >, <, >=, <=` (Comparison) |
| 11 | `not` (Logical NOT) |
| 12 | `and` (Logical AND) |
| 13 (Lowest) | `or` (Logical OR) |

Would you like **more examples** on specific operators? 🚀