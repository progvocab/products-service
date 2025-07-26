### 🧠 ES6 vs JavaScript — What's the Difference?

> ✅ **JavaScript** is the programming language.
> ✅ **ES6** (ECMAScript 2015) is the **6th edition** of the ECMAScript specification — a major update to JavaScript, introducing powerful **syntax improvements and new features**.

So, **ES6 is JavaScript**, just a newer, enhanced version.

---

### 🔍 Quick Summary of Key ES6 Features

| Feature                   | Pre-ES6 JavaScript          | ES6 Syntax                   | Description                              |
| ------------------------- | --------------------------- | ---------------------------- | ---------------------------------------- |
| **Variable Declarations** | `var`                       | `let`, `const`               | Block-scoping, immutability with `const` |
| **Functions**             | Traditional function syntax | Arrow functions `() => {}`   | Shorter, lexical `this`                  |
| **Template Literals**     | `'Hello ' + name`           | `` `Hello ${name}` ``        | Embedded expressions, multiline strings  |
| **Default Parameters**    | `if (x === undefined)`      | `function f(x = 10)`         | Function parameters with defaults        |
| **Destructuring**         | Manual extraction           | `const {x, y} = obj`         | Extract data from arrays/objects easily  |
| **Rest/Spread Operator**  | `arguments`, manual loops   | `...args`, `...arr`          | Concise array/object manipulation        |
| **Modules**               | No native module system     | `import/export`              | Official module support                  |
| **Classes**               | Function-based prototyping  | `class`, `constructor`       | Syntactic sugar for OOP                  |
| **Promises**              | Callback hell               | `Promise`                    | Easier async code                        |
| **Arrow Functions**       | `function()`                | `() => {}`                   | Lexical scoping of `this`                |
| **Iterators/Generators**  | Manual loops                | `function*`, `yield`         | Pause/resume function execution          |
| **Map/Set**               | Plain objects or arrays     | `Map`, `Set`, `WeakMap`      | Improved collections                     |
| **Symbols**               | No equivalent               | `Symbol()`                   | Unique and immutable object keys         |
| **For...of**              | `for`, `forEach()`          | `for (let item of iterable)` | Directly iterates values                 |

---

### 🔥 Code Comparison Examples

#### 1. Variable Declaration

```js
// Pre-ES6
var x = 10;

// ES6
let y = 20;
const z = 30;
```

#### 2. Arrow Functions

```js
// Pre-ES6
function square(n) {
  return n * n;
}

// ES6
const square = n => n * n;
```

#### 3. Template Literals

```js
let name = "Shahbaz";

// Pre-ES6
console.log("Hello, " + name);

// ES6
console.log(`Hello, ${name}`);
```

#### 4. Destructuring

```js
// ES6
const person = { name: "Aamir", age: 30 };
const { name, age } = person;
```

#### 5. Modules

```js
// In moduleA.js
export const pi = 3.14;

// In moduleB.js
import { pi } from './moduleA.js';
```

---

### 🚀 Why It Matters

| Benefit         | How ES6 Helps             |
| --------------- | ------------------------- |
| Cleaner Code    | Less boilerplate          |
| Better Async    | Promises, `async/await`   |
| Safer Variables | `const`, `let` vs. `var`  |
| Easier OOP      | `class` syntax            |
| Modular Apps    | `import/export` structure |

---

Would you like a cheatsheet or hands-on project example with ES6 features?
