Groovy, being a dynamic language built on top of Java, has **many built-in methods** and enhancements (called **GDK methods**) added to standard Java classes. These methods make Groovy scripts more expressive and concise.

Here’s a categorized list of **commonly used built-in functions (GDK methods)** in Groovy:

---

## 🔢 1. **Collection Methods**

Groovy enhances Java collections with many powerful methods.

| Method            | Description                           | Example                                      |
| ----------------- | ------------------------------------- | -------------------------------------------- |
| `each {}`         | Iterates through elements             | `[1, 2, 3].each { println it }`              |
| `collect {}`      | Transforms elements                   | `[1, 2, 3].collect { it * 2 }` → `[2, 4, 6]` |
| `find {}`         | Returns first matching element        | `[1, 2, 3].find { it > 1 }` → `2`            |
| `findAll {}`      | Returns all matching elements         | `[1, 2, 3].findAll { it > 1 }` → `[2, 3]`    |
| `any {}`          | Returns `true` if any element matches | `[1, 2, 3].any { it > 2 }`                   |
| `every {}`        | Returns `true` if all elements match  | `[1, 2, 3].every { it < 4 }`                 |
| `sum()`           | Sum of elements                       | `[1, 2, 3].sum()` → `6`                      |
| `max()` / `min()` | Max/Min element                       | `[1, 5, 3].max()`                            |
| `sort()`          | Sorts the list                        | `[3, 1, 2].sort()`                           |

---

## 🧵 2. **String Methods**

Groovy adds useful string methods beyond Java.

| Method         | Description                         | Example                           |
| -------------- | ----------------------------------- | --------------------------------- |
| `each {}`      | Iterate each character              | `'abc'.each { println it }`       |
| `reverse()`    | Reverse a string                    | `'groovy'.reverse()` → `'yvoorg'` |
| `capitalize()` | Capitalize first letter             | `'groovy'.capitalize()`           |
| `center(n)`    | Centers string within n spaces      | `'abc'.center(7)` → `'  abc  '`   |
| `padLeft(n)`   | Left-pads with spaces               | `'42'.padLeft(5)` → `'   42'`     |
| `padRight(n)`  | Right-pads with spaces              | `'42'.padRight(5)` → `'42   '`    |
| `split()`      | Split by regex or delimiter         | `'a,b,c'.split(',')`              |
| `tokenize()`   | Like `split` but skips empty tokens | `'a,,b'.tokenize(',')`            |
| `contains()`   | Checks for substring                | `'groovy'.contains('oo')`         |
| `replaceAll()` | Replace using regex                 | `'a1b2'.replaceAll(/\d/, '*')`    |

---

## 🔣 3. **Number Methods**

Groovy adds helpers to `int`, `BigDecimal`, etc.

| Method              | Description                  | Example                       |
| ------------------- | ---------------------------- | ----------------------------- |
| `times {}`          | Loop n times                 | `5.times { println it }`      |
| `upto(n) {}`        | Count up from current to `n` | `1.upto(3) { println it }`    |
| `downto(n) {}`      | Count down to `n`            | `3.downto(1) { println it }`  |
| `step(to, step) {}` | Step between numbers         | `1.step(5, 2) { println it }` |

---

## 📁 4. **File Methods**

Groovy adds easy file handling on `File` objects.

| Method         | Description                 | Example                                     |
| -------------- | --------------------------- | ------------------------------------------- |
| `eachLine {}`  | Iterate lines in file       | `new File('a.txt').eachLine { println it }` |
| `text`         | Reads entire file as String | `new File('a.txt').text`                    |
| `write()`      | Overwrite file content      | `file.write('Hello')`                       |
| `append()`     | Append content to file      | `file.append('More')`                       |
| `withReader()` | Opens reader with closure   | `file.withReader { it.readLine() }`         |
| `withWriter()` | Opens writer with closure   | `file.withWriter { it.write("...") }`       |

---

## 🔧 5. **Object & Meta Methods**

| Method           | Description               | Example                         |
| ---------------- | ------------------------- | ------------------------------- |
| `inspect()`      | Returns string of object  | `obj.inspect()`                 |
| `getMetaClass()` | Meta-programming access   | `obj.getMetaClass()`            |
| `invokeMethod()` | Dynamically call a method | `obj.invokeMethod('sayHi', [])` |
| `properties`     | Get map of properties     | `obj.properties['name']`        |

---

## 🧪 6. **Assertions and Scripts**

| Function     | Description                       |
| ------------ | --------------------------------- |
| `assert`     | Assertion (for testing/debugging) |
| `println()`  | Print to stdout                   |
| `sleep(ms)`  | Pause execution                   |
| `evaluate()` | Evaluate Groovy code at runtime   |

---

## 📌 Bonus: Groovy Truth

Groovy uses **truthy/falsy** logic:

```groovy
if ("")      → false
if (null)    → false
if ([])      → false
if ([1, 2])  → true
```

---

Would you like this as a cheatsheet PDF or want examples in a Groovy script file?
