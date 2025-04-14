Java 11 introduced several new features and improvements over Java 8. Here’s a comparison of Java 8 and Java 11, highlighting the features added in Java 11 that were not present in Java 8.

---

## **Key Features in Java 11 (Not Present in Java 8)**

### **1. Local Variable Syntax (`var`) for Lambda Parameters**  
- Java 11 allows the use of `var` in lambda parameters to improve readability and enforce annotations.
- **Java 8:**
  ```java
  (String s) -> s.length();
  ```
- **Java 11:**
  ```java
  (var s) -> s.length();
  ```
  - Useful when applying annotations:
    ```java
    (@NotNull var s) -> s.length();
    ```

---

### **2. `String` API Enhancements**
Several new methods were added to the `String` class:

- **`isBlank()`** → Checks if a string is empty or contains only whitespace.
  ```java
  System.out.println("   ".isBlank()); // true
  ```

- **`strip()`** → Removes leading and trailing whitespace (better than `trim()`).
  ```java
  System.out.println("  hello  ".strip()); // "hello"
  ```

- **`stripLeading()` & `stripTrailing()`** → Remove whitespace only from one side.
  ```java
  System.out.println("  hello  ".stripLeading()); // "hello  "
  System.out.println("  hello  ".stripTrailing()); // "  hello"
  ```

- **`lines()`** → Returns a Stream of lines in a multiline string.
  ```java
  String text = "Line1\nLine2\nLine3";
  text.lines().forEach(System.out::println);
  ```

- **`repeat(int count)`** → Repeats a string `n` times.
  ```java
  System.out.println("Hi ".repeat(3)); // "Hi Hi Hi "
  ```

---

### **3. `Files` API Enhancements**
Java 11 introduced new utility methods in the `Files` class:

- **`Files.readString(Path)`** → Reads file content as a `String`.
  ```java
  String content = Files.readString(Path.of("test.txt"));
  ```

- **`Files.writeString(Path, String)`** → Writes a `String` to a file.
  ```java
  Files.writeString(Path.of("test.txt"), "Hello, Java 11!");
  ```

---

### **4. `Optional` Enhancements**
New methods were added to `Optional`:

- **`isEmpty()`** → Returns `true` if the `Optional` is empty.
  ```java
  Optional<String> opt = Optional.empty();
  System.out.println(opt.isEmpty()); // true
  ```

- **`orElseThrow()`** → Throws `NoSuchElementException` if `Optional` is empty.
  ```java
  String value = opt.orElseThrow();
  ```

---

### **5. HTTP Client (Standardized)**
- Java 8 had the `HttpURLConnection`, but Java 11 introduced a modern HTTP Client in `java.net.http`.
- Supports **asynchronous** calls and **HTTP/2**.

**Example:**
```java
HttpClient client = HttpClient.newHttpClient();
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("https://example.com"))
    .GET()
    .build();

HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
System.out.println(response.body());
```

- Supports **asynchronous requests**:
  ```java
  client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
        .thenApply(HttpResponse::body)
        .thenAccept(System.out::println);
  ```

---

### **6. `Collection.toArray(IntFunction<T[]>)`**
- Java 8 required:
  ```java
  String[] arr = list.toArray(new String[list.size()]);
  ```
- Java 11 allows:
  ```java
  String[] arr = list.toArray(String[]::new);
  ```

---

### **7. New `Collectors` Methods**
- `Collectors.teeing()` → Combines results from two collectors.

**Example:**
```java
import java.util.List;
import java.util.stream.Collectors;

public class TeeingExample {
    public static void main(String[] args) {
        List<Integer> numbers = List.of(1, 2, 3, 4, 5);

        var result = numbers.stream().collect(
            Collectors.teeing(
                Collectors.summingInt(i -> i), // Sum of all numbers
                Collectors.counting(),        // Count of numbers
                (sum, count) -> sum / count   // Average calculation
            )
        );

        System.out.println(result); // Output: 3 (Average)
    }
}
```

---

### **8. `InputStream.readAllBytes()`**
- Reads all bytes from an `InputStream` as a `byte[]`.
  ```java
  byte[] data = inputStream.readAllBytes();
  ```

---

### **9. Running Java Files Without Compilation (`java myfile.java`)**
- In Java 8:
  ```sh
  javac Hello.java
  java Hello
  ```
- In Java 11, you can run `.java` files directly without compiling separately:
  ```sh
  java Hello.java
  ```
  - Useful for scripting and quick testing.

---

### **10. Removal of Deprecated Features (Java 9 & 10 Features)**
- **Java EE and CORBA Modules Removed**:
  - `javax.xml.bind`, `javax.activation`, `javax.annotation`, `javax.transaction`, etc.
- **JavaFX Removed** → Moved to a separate package.

---

### **Key Differences Between Java 8 and Java 11**

| Feature | Java 8 | Java 11 |
|---------|--------|---------|
| `var` for lambda parameters | ❌ | ✅ |
| New `String` methods (`isBlank()`, `strip()`, `repeat()`) | ❌ | ✅ |
| New `Files` API (`readString()`, `writeString()`) | ❌ | ✅ |
| `Optional.isEmpty()` | ❌ | ✅ |
| Modern HTTP Client (`java.net.http`) | ❌ | ✅ |
| `Collection.toArray(T[]::new)` | ❌ | ✅ |
| `Collectors.teeing()` | ❌ | ✅ |
| `InputStream.readAllBytes()` | ❌ | ✅ |
| Running `.java` without compiling (`java MyFile.java`) | ❌ | ✅ |
| Removed Java EE & CORBA Modules | ✅ | ❌ |

---

## **Conclusion**
- **Java 11** provides **improved performance, better APIs, and cleaner code** over Java 8.
- If you are using Java 8, **migrating to Java 11** is beneficial, especially for **modernized APIs** and **long-term support (LTS)**.

Would you like a guide on migrating Java 8 projects to Java 11? Let me know!
