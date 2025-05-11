Let’s walk through a **very simple greedy algorithm use case**:

### **Problem**: Coin Change (Minimum Coins)

> Given a set of coin denominations and a total amount, use the fewest number of coins to make that amount.
> Assume **infinite supply** of each coin and **denominations are canonical** (like 1, 5, 10, 25).

---

### **Example**

* **Denominations**: `[1, 5, 10, 25]`
* **Amount**: `30`

---

### **Step-by-Step Greedy Strategy**

1. **Sort coins in descending order**: `[25, 10, 5, 1]`
2. **Pick the largest coin <= remaining amount**
3. **Subtract its value, record the coin, repeat until 0**

---

### **Steps**

* Start with `amount = 30`
* Pick **25** → amount = `30 - 25 = 5`
* Pick **5** → amount = `5 - 5 = 0`
* Done

**Coins used**: `[25, 5]`

---

### **Python Code**

```python
def min_coins_greedy(coins, amount):
    coins.sort(reverse=True)  # Step 1
    result = []

    for coin in coins:
        while amount >= coin:
            amount -= coin       # Step 2 & 3
            result.append(coin)  # Step 4
        if amount == 0:
            break

    return result
```

```python
# Usage
coins = [1, 5, 10, 25]
amount = 30
print(min_coins_greedy(coins, amount))  # Output: [25, 5]
```

---

### **Important Note**

Greedy works **only when denominations are canonical** (like U.S. coins).
In some cases (e.g. `[1, 3, 4]`, amount = `6`), greedy fails to give the minimum coins.

Would you like an example where greedy fails compared to dynamic programming?
