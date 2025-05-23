This is a classic **dynamic programming** problem that follows the **Fibonacci pattern**.

---

### **Problem Breakdown**:

* To reach step `n`, you could have come from:

  * Step `n-1` (taking 1 step)
  * Step `n-2` (taking 2 steps)
* So the total ways to reach step `n` is:

  ```
  ways(n) = ways(n-1) + ways(n-2)
  ```

---

### **Python Solution**:

```python
def climb_stairs(n):
    if n <= 2:
        return n

    a, b = 1, 2  # base cases: 1 way to reach step 1, 2 ways to reach step 2
    for i in range(3, n + 1):
        a, b = b, a + b  # update the number of ways dynamically

    return b

# Example usage
n = 5
print(f"Number of distinct ways to climb {n} steps:", climb_stairs(n))
```

---

### **Output**:

```
Number of distinct ways to climb 5 steps: 8
```

*Explanation: The sequence is 1, 2, 3, 5, 8... like Fibonacci.*

Let me know if you'd like a recursive or memoized version too.
