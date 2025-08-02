### 🧮 Procedural Programming Paradigm

**Procedural Programming** is a **structured, imperative** programming paradigm based on the concept of **procedure calls**, where programs are built as a sequence of **instructions** grouped into **functions** or **procedures**.

It is one of the earliest and most fundamental paradigms and forms the basis of many modern programming languages.

---

## 🔑 Key Concepts

| Concept                  | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| **Procedures/Functions** | Named blocks of code that perform a task and can be reused    |
| **Sequence**             | Code executes step-by-step in order                           |
| **Selection**            | Uses conditionals (if/else, switch) for decision-making       |
| **Iteration**            | Repeats actions via loops (for, while)                        |
| **Global/Local State**   | Data is stored in variables that can be accessed or modified  |
| **Top-Down Design**      | Programs are broken down into smaller, manageable subroutines |

---

## ✅ Example in C (Classic Procedural Language)

```c
#include <stdio.h>

void printSum(int a, int b) {
    int sum = a + b;
    printf("Sum: %d\n", sum);
}

int main() {
    int x = 10, y = 20;
    printSum(x, y);  // Call the procedure
    return 0;
}
```

---

## 📌 Characteristics

| Feature                | Procedural Programming                                      |
| ---------------------- | ----------------------------------------------------------- |
| **Modularity**         | Achieved using procedures/functions                         |
| **Code Reusability**   | Code blocks can be reused in different parts of the program |
| **Control Flow**       | Based on sequence, selection, and iteration                 |
| **State**              | Usually uses shared, mutable state                          |
| **Data and Functions** | Kept separate (unlike OOP)                                  |

---

## 📚 Languages that support it

| Language                     | Notes                                                     |
| ---------------------------- | --------------------------------------------------------- |
| **C**                        | Pure procedural                                           |
| **Pascal**                   | Structured procedural                                     |
| **Fortran**                  | One of the first procedural languages                     |
| **BASIC**                    | Beginner-oriented procedural                              |
| **Python, Java, JavaScript** | Support procedural programming along with other paradigms |

---

## ✅ Advantages

| Advantage                           | Explanation                                                           |
| ----------------------------------- | --------------------------------------------------------------------- |
| **Simple & Straightforward**        | Easy to understand for small tasks                                    |
| **Efficient for Performance**       | Less abstraction than OOP or FP                                       |
| **Well-Suited for Low-Level Tasks** | Ideal for systems-level programming like embedded systems, OS kernels |
| **Easier to Debug Small Programs**  | Because execution is linear and predictable                           |

---

## ❌ Disadvantages

| Drawback                | Why?                                                                          |
| ----------------------- | ----------------------------------------------------------------------------- |
| **Poor Scalability**    | Difficult to manage large codebases with many global variables and procedures |
| **Less Reusability**    | Lacks the extensibility and inheritance of OOP                                |
| **Tight Coupling**      | Functions often rely on shared global state                                   |
| **No Data Abstraction** | Data and logic are not bound together                                         |

---

## 🔁 Procedural vs Other Paradigms

| Feature           | Procedural                | Object-Oriented                  | Functional                  |
| ----------------- | ------------------------- | -------------------------------- | --------------------------- |
| **Unit of Code**  | Procedure                 | Object (class + methods)         | Function                    |
| **Data Handling** | Global/Local vars         | Encapsulated                     | Immutable                   |
| **State**         | Mutable                   | Mutable (encapsulated)           | Immutable                   |
| **Reusability**   | Moderate (via procedures) | High (inheritance, polymorphism) | High (function composition) |

---

## 🏁 Summary

> **Procedural Programming** is ideal for tasks where performance and simplicity are key. It works well for small to medium applications, systems programming, and environments where abstraction is unnecessary or undesired.

Would you like to see the same logic implemented in **procedural**, **OOP**, and **functional** styles for comparison?
