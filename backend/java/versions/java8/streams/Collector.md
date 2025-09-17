In Java, the `Collector` interface from the `java.util.stream` package is used to accumulate elements from a stream into a final result, typically used in the `Stream.collect()` terminal operation.

---
### As per Javadoc 
A Collector is specified by four functions that work together to accumulate entries into a mutable result container, and optionally perform a final transform on the result. They are:
```text

```


- creation of a new result container (supplier())
- incorporating a new data element into a result container (accumulator())
- combining two result containers into one (combiner())
- performing an optional final transform on the container (finisher())
- Collectors also have a set of characteristics, such as Collector.Characteristics.CONCURRENT, that provide hints that can be used by a reduction implementation to provide better performance.
```text
```
---

### **When to implement a custom Collector?**
You might need a custom `Collector` when:

- You want a non-standard accumulation logic.
- You want to collect elements into a custom container or data structure.
- Built-in collectors like `Collectors.toList()`, `toSet()`, `toMap()` don’t meet your needs.

---

### **Example Scenario:**
**Goal:** Group strings by their length and concatenate them with commas.

Input:
```java
List<String> words = List.of("apple", "bat", "cat", "banana", "dog");
```

Desired Output (Map):
```java
{
  3 = "bat,cat,dog",
  5 = "apple",
  6 = "banana"
}
```

This isn't directly achievable with built-in collectors, so we create a **custom Collector**.

---

### **Custom Collector Implementation:**

```java
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class GroupingByLengthCollector implements Collector<
        String,                             // Input type
        Map<Integer, StringJoiner>,         // Accumulation type
        Map<Integer, String>                // Result type
    > {

    @Override
    public Supplier<Map<Integer, StringJoiner>> supplier() {
        return HashMap::new;
    }

    @Override
    public BiConsumer<Map<Integer, StringJoiner>, String> accumulator() {
        return (map, word) -> {
            int length = word.length();
            map.computeIfAbsent(length, k -> new StringJoiner(",")).add(word);
        };
    }

    @Override
    public BinaryOperator<Map<Integer, StringJoiner>> combiner() {
        return (map1, map2) -> {
            map2.forEach((k, v) -> map1.merge(k, v, StringJoiner::merge));
            return map1;
        };
    }

    @Override
    public Function<Map<Integer, StringJoiner>, Map<Integer, String>> finisher() {
        return joinerMap -> {
            Map<Integer, String> result = new HashMap<>();
            joinerMap.forEach((k, v) -> result.put(k, v.toString()));
            return result;
        };
    }

    @Override
    public Set<Characteristics> characteristics() {
        return Collections.emptySet(); // Not CONCURRENT or UNORDERED
    }
}
```

---

### **Using the Custom Collector:**

```java
public class CustomCollectorDemo {
    public static void main(String[] args) {
        List<String> words = List.of("apple", "bat", "cat", "banana", "dog");

        Map<Integer, String> result = words.stream()
                .collect(new GroupingByLengthCollector());

        result.forEach((length, concat) ->
            System.out.println("Length " + length + ": " + concat));
    }
}
```

---

### **Output:**
```
Length 5: apple
Length 3: bat,cat,dog
Length 6: banana
```

---

Would you like a version of this collector using `Collector.of(...)` instead of a full class implementation?


Here is a simplified version of the **custom collector** using `Collector.of(...)`, which avoids creating a full class:

---

### **Collector.of(...) Implementation:**

```java
import java.util.*;
import java.util.stream.*;
import java.util.function.*;

public class CustomCollectorWithOf {
    public static void main(String[] args) {
        List<String> words = List.of("apple", "bat", "cat", "banana", "dog");

        Map<Integer, String> result = words.stream().collect(
            Collector.of(
                HashMap::new, // supplier
                (map, word) -> {
                    int len = word.length();
                    map.computeIfAbsent(len, k -> new StringJoiner(",")).add(word);
                },
                (map1, map2) -> {
                    map2.forEach((k, v) -> map1.merge(k, v, StringJoiner::merge));
                    return map1;
                },
                (joinerMap) -> {
                    Map<Integer, String> finalMap = new HashMap<>();
                    joinerMap.forEach((k, v) -> finalMap.put(k, v.toString()));
                    return finalMap;
                }
            )
        );

        result.forEach((length, concat) ->
            System.out.println("Length " + length + ": " + concat));
    }
}
```

---

### **Output:**
```
Length 5: apple
Length 3: bat,cat,dog
Length 6: banana
```

---

This `Collector.of(...)` version is more concise and commonly used for one-off collector definitions inside stream operations.

Would you like to implement a custom collector for a different data structure or operation next?


Great! Let’s now create a **custom collector** for a different operation. Here’s a useful scenario:

---

### **Use Case**:  
You want to **collect a list of strings** into a `Map<Character, List<String>>` where each string is grouped by its **first character** (case insensitive).

---

### **Custom Collector with `Collector.of(...)`**:

```java
import java.util.*;
import java.util.stream.*;
import java.util.function.*;

public class GroupByFirstCharCollector {
    public static void main(String[] args) {
        List<String> words = List.of("Apple", "ant", "Ball", "bat", "banana", "Cat", "camel");

        Map<Character, List<String>> grouped = words.stream().collect(
            Collector.of(
                HashMap::new, // supplier
                (map, word) -> {
                    char key = Character.toLowerCase(word.charAt(0));
                    map.computeIfAbsent(key, k -> new ArrayList<>()).add(word);
                },
                (map1, map2) -> {
                    map2.forEach((k, v) -> map1.merge(k, v, (list1, list2) -> {
                        list1.addAll(list2);
                        return list1;
                    }));
                    return map1;
                }
            )
        );

        grouped.forEach((k, v) -> System.out.println(k + ": " + v));
    }
}
```

---

### **Output** (varies by order):
```
a: [Apple, ant]
b: [Ball, bat, banana]
c: [Cat, camel]
```

---

### Why use a custom collector?
- You need control over how items are accumulated, combined, or finalized.
- Especially useful when `Collectors.groupingBy(...)` doesn’t support your special merging logic or key transformation.

Would you like one with a **custom finisher**, **immutability**, or maybe accumulate a different structure (like a `TreeMap` or a `Set`)?
