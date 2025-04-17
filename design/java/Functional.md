Functional programming in Java was greatly enhanced starting with **Java 8**, which introduced **lambda expressions**, **functional interfaces**, and the **Stream API**. Here's an overview of functional programming concepts in Java with examples:

---

### **1. Key Concepts of Functional Programming**

| Concept                     | Java Feature                            |
|----------------------------|------------------------------------------|
| Functions as First-Class   | Lambda Expressions, Method References    |
| Immutability               | `final`, Avoid mutating state            |
| Higher-Order Functions     | Passing lambdas as parameters            |
| Pure Functions             | No side effects, return same output      |
| Declarative Style          | Streams, `map()`, `filter()`, `reduce()` |

---

### **2. Lambda Expressions**

```java
// Traditional way
Runnable r1 = new Runnable() {
    public void run() {
        System.out.println("Running...");
    }
};

// Functional way
Runnable r2 = () -> System.out.println("Running...");
```

---

### **3. Functional Interfaces**

A functional interface has only one abstract method.

```java
@FunctionalInterface
interface MathOperation {
    int operate(int a, int b);
}

MathOperation add = (a, b) -> a + b;
System.out.println(add.operate(5, 3)); // Output: 8
```

---

### **4. Streams API**

```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");

List<String> filtered = names.stream()
    .filter(name -> name.startsWith("A"))
    .map(String::toUpperCase)
    .collect(Collectors.toList());

System.out.println(filtered); // [ALICE]
```

---

### **5. Method References**

```java
List<String> list = Arrays.asList("a", "b", "c");
list.forEach(System.out::println); // Instead of s -> System.out.println(s)
```

---

### **6. Optional for Null Safety**

```java
Optional<String> name = Optional.of("John");
name.ifPresent(System.out::println); // John
```

---

### **7. `Predicate`, `Function`, `Consumer`, `Supplier`**

```java
Predicate<String> isEmpty = String::isEmpty;
Function<Integer, String> toString = Object::toString;
Consumer<String> printer = System.out::println;
Supplier<String> supplier = () -> "Hello";
```

---

Would you like examples on composing functions or custom functional interfaces?



### Functional 
Functional programming in Java emphasizes using pure functions and immutable data to create more concise, readable, and maintainable code. Java 8 introduced features like lambda expressions, streams, and functional interfaces, making it easier to incorporate functional programming principles into Java code. [1, 2, 3]  
Key Concepts in Functional Programming: [1, 4]  

- Pure Functions: Functions that always return the same output for a given input and don't have side effects (e.g., modifying global variables or external state). [1, 4]  
- Immutability: Data that cannot be changed after it's created. [1, 4]  
- Higher-Order Functions: Functions that can take other functions as arguments or return them as values. [1, 5]  
- First-Class Functions: Functions that can be treated like any other data type (passed as arguments, returned from functions, etc.). [1, 5]  
- Function Composition: Combining multiple functions to create a new function. [1, 5]  

### Benefits of Functional Programming in Java: [1, 2]  

- Conciseness and Readability: Functional code can be more compact and easier to understand due to the declarative style. 
- Improved Testability: Pure functions are easier to test because they don't rely on external state. 
- Reduced Bugs: Immutability and avoiding side effects can help prevent common programming errors. 
- Concurrency: Functional programming can be well-suited for concurrent and parallel programming due to its reliance on immutable data and pure functions. [1, 2]  

Java Features for Functional Programming: [3, 6]  

- Lambda Expressions: Concise, anonymous functions that can be used to implement functional interfaces. [3, 6]  
- Functional Interfaces: Interfaces with a single abstract method that can be used with lambda expressions. [3, 7]  
- Streams: A sequence of elements that support various functional operations like filtering, mapping, and reducing. [1, 2]  
- Optional: A container object that may or may not contain a value, helping to avoid null pointer exceptions. [1, 2]  

### Example: 
``` java
// Imperative approach
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
List<Integer> evenNumbers = new ArrayList<>();
for (int number : numbers) {
    if (number % 2 == 0) {
        evenNumbers.add(number);
    }
}

// Functional approach using streams
List<Integer> evenNumbers = numbers.stream()
        .filter(number -> number % 2 == 0)
        .collect(Collectors.toList());

```

Generative AI is experimental.

[1] https://www.niit.com/india/Functional-Programming-in-Java-8-Tips-and-Tricks[2] https://medium.com/javajams/master-functional-programming-in-java-a-practical-guide-cade78f1d0b0[3] https://medium.com/@dulanjayasandaruwan1998/an-introduction-to-functional-programming-in-java-3c7f3493affc[4] https://www.scaler.com/topics/java/functional-programming-in-java/[5] https://jenkov.com/tutorials/java-functional-programming/index.html[6] https://medium.com/@AlexanderObregon/lambda-expressions-and-functional-programming-in-java-ce81613380a5[7] https://www.geekster.in/articles/function-interface-in-java/[-] https://www.bytezonex.com/archives/yutSvA0D.html
