In Java, `System.out.printf()` is used to print **formatted output**, similar to C-style formatting. It’s part of the `PrintStream` class and supports a **rich set of format specifiers** for different data types.

---

## **Basic Syntax**

```java
System.out.printf(String format, Object... args);
```

- `format`: a string containing text and **format specifiers**
- `args`: values that replace specifiers

---

## **Common Format Specifiers**

| Format | Type         | Example             |
|--------|--------------|---------------------|
| `%d`   | Integer      | `System.out.printf("%d", 10);`         |
| `%f`   | Float/Double | `System.out.printf("%.2f", 3.1415);`   |
| `%s`   | String       | `System.out.printf("%s", "Java");`     |
| `%c`   | Character    | `System.out.printf("%c", 'A');`        |
| `%b`   | Boolean      | `System.out.printf("%b", true);`       |
| `%n`   | Newline      | `System.out.printf("Line 1%nLine 2");` |
| `%x` / `%X` | Hex integer | `System.out.printf("%x", 255);`   |

---

## **Width and Precision**

| Feature      | Example                       | Output       |
|--------------|-------------------------------|--------------|
| Width        | `"%10s"`                      | Right-aligned string |
| Left-align   | `"%-10s"`                     | Left-aligned |
| Precision    | `"%.2f"`                      | 2 decimal places |
| Width + Prec | `"%8.2f"`                     | Total 8 chars, 2 decimals |

---

### **Examples**

```java
System.out.printf("Hello, %s!%n", "World");       // Hello, World!
System.out.printf("Int: %d%n", 42);               // Int: 42
System.out.printf("Float: %.2f%n", 3.14159);      // Float: 3.14
System.out.printf("Char: %c%n", 'J');             // Char: J
System.out.printf("Bool: %b%n", null);            // Bool: false
System.out.printf("Hex: %x%n", 255);              // Hex: ff
```

---

## **Flags in Format Specifiers**

| Flag | Description             | Example         | Result           |
|------|-------------------------|------------------|------------------|
| `-`  | Left justify            | `"%-10s"`       | `"abc       "`   |
| `+`  | Always show sign        | `"%+d"`         | `+10`            |
| `0`  | Pad with zeros          | `"%05d"`        | `00042`          |
| `,`  | Use locale-specific grouping | `"%,d"`   | `1,000,000`      |
| `(`  | Enclose negative numbers in parentheses | `"%(d"` | `(1000)` |

---

## **Advanced Specifiers**

| Specifier   | Description                |
|-------------|----------------------------|
| `%e` / `%E` | Scientific notation        |
| `%g` / `%G` | Uses `%f` or `%e` based on value |
| `%t`        | Date/time (e.g., `%tY`, `%tB`) |
| `%n`        | Platform-independent newline |

---

## **Date/Time Formatting**

```java
Date now = new Date();
System.out.printf("Year: %tY%n", now);    // e.g., Year: 2025
System.out.printf("Month: %tB%n", now);   // e.g., Month: April
System.out.printf("Time: %tT%n", now);    // e.g., Time: 14:23:45
```

---

Would you like a cheat sheet or examples formatted as a table output?