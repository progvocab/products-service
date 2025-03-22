The `java.util.function` package, introduced in Java 8, provides a set of functional interfaces that are fundamental to functional programming in Java, especially with lambda expressions and the Stream API. These interfaces represent various function types and are used extensively in modern Java for operations like mapping, filtering, and reducing data.

Below, I'll explain each interface in the package, categorized by their purpose, and provide code examples for clarity.

---

### Categories of Functional Interfaces
1. **Consumers**: Accept input(s) and perform an action (no return value).
2. **Suppliers**: Provide a result (no input).
3. **Functions**: Transform input(s) into an output.
4. **Predicates**: Test input(s) and return a boolean.
5. **Operators**: Special Functions that return the same type as input(s).

---

### 1. Consumers
These accept arguments and return no result (void).

#### `Consumer<T>`
- **Purpose**: Represents an operation that accepts a single input and returns no result.
- **Method**: `void accept(T t)`
```java
import java.util.function.Consumer;

public class Main {
    public static void main(String[] args) {
        Consumer<String> printer = s -> System.out.println(s);
        printer.accept("Hello, World!");  // Output: Hello, World!
    }
}
```

#### `BiConsumer<T, U>`
- **Purpose**: Accepts two inputs and returns no result.
- **Method**: `void accept(T t, U u)`
```java
import java.util.function.BiConsumer;

public class Main {
    public static void main(String[] args) {
        BiConsumer<String, Integer> printPair = (s, i) -> System.out.println(s + ": " + i);
        printPair.accept("Age", 25);  // Output: Age: 25
    }
}
```

---

### 2. Suppliers
These provide a result without taking input.

#### `Supplier<T>`
- **Purpose**: Supplies a value of type T.
- **Method**: `T get()`
```java
import java.util.function.Supplier;

public class Main {
    public static void main(String[] args) {
        Supplier<String> greeter = () -> "Hello!";
        System.out.println(greeter.get());  // Output: Hello!
    }
}
```

#### `BooleanSupplier`, `DoubleSupplier`, `IntSupplier`, `LongSupplier`
- **Purpose**: Primitive-specific suppliers returning `boolean`, `double`, `int`, or `long`.
- **Method**: `<type> getAs<Type>()`
```java
import java.util.function.IntSupplier;

public class Main {
    public static void main(String[] args) {
        IntSupplier randomInt = () -> (int) (Math.random() * 10);
        System.out.println(randomInt.getAsInt());  // Output: Random number 0-9
    }
}
```

---

### 3. Functions
These transform input(s) into an output.

#### `Function<T, R>`
- **Purpose**: Takes an input of type T and returns a result of type R.
- **Method**: `R apply(T t)`
```java
import java.util.function.Function;

public class Main {
    public static void main(String[] args) {
        Function<String, Integer> length = s -> s.length();
        System.out.println(length.apply("Java"));  // Output: 4
    }
}
```

#### `BiFunction<T, U, R>`
- **Purpose**: Takes two inputs and returns a result.
- **Method**: `R apply(T t, U u)`
```java
import java.util.function.BiFunction;

public class Main {
    public static void main(String[] args) {
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        System.out.println(add.apply(3, 4));  // Output: 7
    }
}
```

#### `ToDoubleFunction<T>`, `ToIntFunction<T>`, `ToLongFunction<T>`
- **Purpose**: Converts an input to a primitive `double`, `int`, or `long`.
- **Method**: `<type> applyAs<Type>(T t)`
```java
import java.util.function.ToIntFunction;

public class Main {
    public static void main(String[] args) {
        ToIntFunction<String> strLength = s -> s.length();
        System.out.println(strLength.applyAsInt("Hello"));  // Output: 5
    }
}
```

#### `DoubleFunction<R>`, `IntFunction<R>`, `LongFunction<R>`
- **Purpose**: Takes a primitive input and returns a result of type R.
- **Method**: `R apply(<type> value)`
```java
import java.util.function.IntFunction;

public class Main {
    public static void main(String[] args) {
        IntFunction<String> intToStr = i -> "Number: " + i;
        System.out.println(intToStr.apply(42));  // Output: Number: 42
    }
}
```

#### `DoubleToIntFunction`, `DoubleToLongFunction`, `IntToDoubleFunction`, `IntToLongFunction`, `LongToDoubleFunction`, `LongToIntFunction`
- **Purpose**: Converts between primitive types.
- **Method**: `<type> applyAs<Type>(<type> value)`
```java
import java.util.function.IntToDoubleFunction;

public class Main {
    public static void main(String[] args) {
        IntToDoubleFunction half = i -> i / 2.0;
        System.out.println(half.applyAsDouble(5));  // Output: 2.5
    }
}
```

---

### 4. Predicates
These test conditions and return a boolean.

#### `Predicate<T>`
- **Purpose**: Tests an input and returns a boolean.
- **Method**: `boolean test(T t)`
```java
import java.util.function.Predicate;

public class Main {
    public static void main(String[] args) {
        Predicate<Integer> isEven = n -> n % 2 == 0;
        System.out.println(isEven.test(4));  // Output: true
        System.out.println(isEven.test(5));  // Output: false
    }
}
```

#### `BiPredicate<T, U>`
- **Purpose**: Tests two inputs and returns a boolean.
- **Method**: `boolean test(T t, U u)`
```java
import java.util.function.BiPredicate;

public class Main {
    public static void main(String[] args) {
        BiPredicate<String, Integer> lengthCheck = (s, i) -> s.length() > i;
        System.out.println(lengthCheck.test("Java", 3));  // Output: true
    }
}
```

#### `DoublePredicate`, `IntPredicate`, `LongPredicate`
- **Purpose**: Tests a primitive value and returns a boolean.
- **Method**: `boolean test(<type> value)`
```java
import java.util.function.IntPredicate;

public class Main {
    public static void main(String[] args) {
        IntPredicate isPositive = i -> i > 0;
        System.out.println(isPositive.test(5));   // Output: true
        System.out.println(isPositive.test(-1));  // Output: false
    }
}
```

---

### 5. Operators
Specialized Functions where input and output types match.

#### `UnaryOperator<T>`
- **Purpose**: Takes one argument and returns a result of the same type.
- **Method**: `T apply(T t)`
```java
import java.util.function.UnaryOperator;

public class Main {
    public static void main(String[] args) {
        UnaryOperator<Integer> square = n -> n * n;
        System.out.println(square.apply(5));  // Output: 25
    }
}
```

#### `BinaryOperator<T>`
- **Purpose**: Takes two arguments of the same type and returns a result of that type.
- **Method**: `T apply(T t1, T t2)`
```java
import java.util.function.BinaryOperator;

public class Main {
    public static void main(String[] args) {
        BinaryOperator<Integer> max = (a, b) -> a > b ? a : b;
        System.out.println(max.apply(3, 7));  // Output: 7
    }
}
```

#### `DoubleUnaryOperator`, `IntUnaryOperator`, `LongUnaryOperator`
- **Purpose**: Unary operation on a primitive type, returning the same type.
- **Method**: `<type> applyAs<Type>(<type> operand)`
```java
import java.util.function.IntUnaryOperator;

public class Main {
    public static void main(String[] args) {
        IntUnaryOperator increment = i -> i + 1;
        System.out.println(increment.applyAsInt(10));  // Output: 11
    }
}
```

#### `DoubleBinaryOperator`, `IntBinaryOperator`, `LongBinaryOperator`
- **Purpose**: Binary operation on two primitive values, returning the same type.
- **Method**: `<type> applyAs<Type>(<type> left, <type> right)`
```java
import java.util.function.IntBinaryOperator;

public class Main {
    public static void main(String[] args) {
        IntBinaryOperator multiply = (a, b) -> a * b;
        System.out.println(multiply.applyAsInt(4, 5));  // Output: 20
    }
}
```

---

### Key Points
- **Primitive Variants**: Many interfaces have primitive-specific versions (e.g., `IntFunction`, `DoublePredicate`) to avoid boxing/unboxing overhead.
- **Default Methods**: Some interfaces (e.g., `Function`, `Predicate`) provide default methods like `andThen`, `compose`, `and`, `or`, `negate` for function composition.
- **Usage**: These are heavily used with Streams, lambdas, and method references.

Example with Stream:
```java
import java.util.Arrays;
import java.util.function.Predicate;

public class Main {
    public static void main(String[] args) {
        Predicate<Integer> isOdd = n -> n % 2 != 0;
        int[] numbers = {1, 2, 3, 4, 5};
        Arrays.stream(numbers)
              .filter(isOdd)
              .forEach(System.out::println);  // Output: 1, 3, 5
    }
}
```

This covers all interfaces in `java.util.function`. Let me know if you'd like deeper examples or clarification on any specific interface!