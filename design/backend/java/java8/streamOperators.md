In Java's **Streams API**, operations are categorized into two types: **intermediate operators** and **terminal operators**. These operators allow developers to process collections or streams of data in a functional and declarative style. Here's a breakdown of each type:

---

### **1. Intermediate Operators**
Intermediate operators transform a stream into another stream. They are **lazy**, meaning they do not process data immediately but instead build a pipeline of operations. Actual computation happens only when a terminal operation is invoked.

#### **Key Characteristics**:
- They are lazy and do not execute until a terminal operator is called.
- They return another stream, allowing chaining of operations.
- Used for transforming or filtering data.

#### **Examples**:
1. **`filter()`**: Filters elements based on a predicate.
   ```java
   Stream<Integer> stream = Stream.of(1, 2, 3, 4, 5).filter(n -> n % 2 == 0);
   ```

2. **`map()`**: Transforms each element in the stream.
   ```java
   Stream<String> stream = Stream.of("a", "b", "c").map(String::toUpperCase);
   ```

3. **`flatMap()`**: Flattens nested streams into a single stream.
   ```java
   Stream<String> stream = Stream.of(Arrays.asList("a", "b"), Arrays.asList("c", "d"))
                                 .flatMap(List::stream);
   ```

4. **`distinct()`**: Removes duplicate elements.
   ```java
   Stream<Integer> stream = Stream.of(1, 2, 2, 3, 3, 4).distinct();
   ```

5. **`sorted()`**: Sorts the elements in natural order or using a comparator.
   ```java
   Stream<Integer> stream = Stream.of(5, 2, 1, 4).sorted();
   ```

6. **`limit()`**: Limits the size of the stream.
   ```java
   Stream<Integer> stream = Stream.of(1, 2, 3, 4, 5).limit(3);
   ```

7. **`skip()`**: Skips the first n elements.
   ```java
   Stream<Integer> stream = Stream.of(1, 2, 3, 4, 5).skip(2);
   ```

---

### **2. Terminal Operators**
Terminal operators trigger the execution of the stream pipeline and produce a result. They are **eager**, meaning they process all elements in the stream immediately.

#### **Key Characteristics**:
- They consume the stream, so it cannot be reused after a terminal operation.
- Produce a result such as a collection, a value, or a side effect.
- Without a terminal operator, intermediate operations are not executed.

#### **Examples**:
1. **`collect()`**: Collects elements into a collection or a result.
   ```java
   List<Integer> list = Stream.of(1, 2, 3, 4).collect(Collectors.toList());
   ```

2. **`forEach()`**: Performs an action for each element.
   ```java
   Stream.of(1, 2, 3).forEach(System.out::println);
   ```

3. **`toArray()`**: Converts the stream to an array.
   ```java
   Integer[] array = Stream.of(1, 2, 3).toArray(Integer[]::new);
   ```

4. **`reduce()`**: Combines elements of the stream into a single result.
   ```java
   int sum = Stream.of(1, 2, 3).reduce(0, Integer::sum);
   ```

5. **`count()`**: Counts the number of elements in the stream.
   ```java
   long count = Stream.of(1, 2, 3).count();
   ```

6. **`findFirst()`**: Returns the first element in the stream (if any).
   ```java
   Optional<Integer> first = Stream.of(1, 2, 3).findFirst();
   ```

7. **`findAny()`**: Returns any element in the stream (useful in parallel streams).
   ```java
   Optional<Integer> any = Stream.of(1, 2, 3).findAny();
   ```

8. **`anyMatch()`**: Checks if any elements match a predicate.
   ```java
   boolean exists = Stream.of(1, 2, 3).anyMatch(n -> n > 2);
   ```

9. **`allMatch()`**: Checks if all elements match a predicate.
   ```java
   boolean all = Stream.of(1, 2, 3).allMatch(n -> n > 0);
   ```

10. **`noneMatch()`**: Checks if no elements match a predicate.
    ```java
    boolean none = Stream.of(1, 2, 3).noneMatch(n -> n < 0);
    ```

11. **`max()` and `min()`**: Finds the maximum or minimum element using a comparator.
    ```java
    Optional<Integer> max = Stream.of(1, 2, 3).max(Integer::compare);
    ```

---

### **Comparison of Intermediate vs. Terminal Operators**

| **Aspect**                | **Intermediate Operators**                             | **Terminal Operators**                         |
|---------------------------|-------------------------------------------------------|-----------------------------------------------|
| **Purpose**               | Transform or filter the stream.                       | Produce a result or side effect from the stream. |
| **Execution**             | Lazy (pipeline builds but no processing happens).     | Eager (triggers execution of the pipeline).    |
| **Return Type**           | Returns a new stream for further chaining.            | Returns a result (e.g., collection, value, or void). |
| **Examples**              | `filter()`, `map()`, `sorted()`, `distinct()`.        | `collect()`, `reduce()`, `forEach()`, `count()`. |
| **Reuse of Stream**       | Stream can still be used after the operation.         | Stream is consumed and cannot be reused.       |

---

### **Example Combining Both**
Here’s an example showing how intermediate and terminal operators work together:

```java
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class StreamExample {
    public static void main(String[] args) {
        // Intermediate operations: filter, map
        // Terminal operation: collect
        List<String> result = Stream.of("apple", "banana", "cherry", "date")
                                    .filter(s -> s.startsWith("b"))
                                    .map(String::toUpperCase)
                                    .collect(Collectors.toList());

        System.out.println(result); // Output: [BANANA]
    }
}
```

---

### **Key Takeaways**
1. **Intermediate operators** build a pipeline of transformations, while **terminal operators** trigger processing.
2. Without a terminal operator, intermediate operations are not executed.
3. Streams provide a functional programming approach to handle data efficiently and declaratively. 

Let me know if you'd like more examples or clarification!