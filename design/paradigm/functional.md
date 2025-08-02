### 🧮 Functional Programming (FP) Paradigm

**Functional Programming** is a **declarative** programming paradigm where computation is treated as the **evaluation of mathematical functions**. It emphasizes **immutability**, **pure functions**, and **stateless computation**, avoiding changing-state and mutable data.

---

## 🧠 Key Concepts

| Concept                      | Description                                                             |
| ---------------------------- | ----------------------------------------------------------------------- |
| **Pure Functions**           | Same input always produces same output; no side effects                 |
| **Immutability**             | Variables are never modified after they're assigned                     |
| **First-Class Functions**    | Functions are treated like values (can be passed/stored)                |
| **Higher-Order Functions**   | Functions that take or return other functions                           |
| **Function Composition**     | Combining functions to build more complex ones                          |
| **Recursion**                | Replaces loops; functions call themselves                               |
| **Referential Transparency** | Expressions can be replaced with their values without changing behavior |

---

## ✅ Example in Java (using Java 8+ Streams)

```java
import java.util.List;

public class FPExample {
    public static void main(String[] args) {
        List<Integer> nums = List.of(1, 2, 3, 4, 5);

        int sumOfSquares = nums.stream()
                               .map(x -> x * x)        // pure function
                               .reduce(0, Integer::sum); // function composition

        System.out.println(sumOfSquares);  // Output: 55
    }
}
```

### 🔍 Breakdown:

* `map` transforms each element
* `reduce` aggregates the result
* No mutation of state; no side effects

---

## 🧰 Languages That Support FP

| Language       | Functional Support                          |
| -------------- | ------------------------------------------- |
| **Haskell**    | Pure functional                             |
| **Scala**      | Functional + OOP                            |
| **Elixir**     | Functional                                  |
| **JavaScript** | Supports FP                                 |
| **Python**     | Supports FP (via `map`, `filter`, `lambda`) |
| **Java 8+**    | Streams API introduces FP concepts          |

---

## 🌍 Real-World Analogy

Imagine a **vending machine**:

* You input ₹10 → You get a snack → Always the same output → No state stored between requests.
* That’s like a **pure function**: `f(10) = "snack"`

---

## ⚙️ Benefits of FP

| Benefit                    | Description                               |
| -------------------------- | ----------------------------------------- |
| **Predictability**         | Pure functions are easier to reason about |
| **Testability**            | Functions can be tested independently     |
| **Concurrency-safe**       | No shared state or side effects           |
| **Modular and Composable** | Functions are easy to reuse and combine   |

---

## 🚫 Challenges

| Drawback                   | Why?                                                                      |
| -------------------------- | ------------------------------------------------------------------------- |
| **Learning curve**         | Concepts like monads, currying, etc. are non-intuitive for OOP developers |
| **Performance overhead**   | Due to recursion and immutability                                         |
| **Verbose error messages** | In statically typed FP languages like Haskell                             |
| **Hard to manage state**   | Since state must be passed explicitly                                     |

---

## 💡 Functional vs Object-Oriented

| Aspect             | Functional Programming     | Object-Oriented Programming |
| ------------------ | -------------------------- | --------------------------- |
| **Focus**          | Functions and immutability | Objects and state           |
| **Data**           | Immutable                  | Mutable                     |
| **Composition**    | Function composition       | Inheritance                 |
| **State Handling** | Explicit                   | Encapsulated                |

---

Would you like examples in a specific language like **JavaScript**, **Haskell**, or **Python**? Or a comparison with **Reactive Programming**?
