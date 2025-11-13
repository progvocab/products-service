

### ⚙️ **Prop Drilling (Definition):**

**Prop drilling** is when you pass data through **multiple intermediate components** that don’t need it, just to get it to a **deeply nested child**.

---

###  **Example:**

```jsx
function App() {
  const user = "Alice";
  return <Parent user={user} />;
}

function Parent({ user }) {
  return <Child user={user} />;
}

function Child({ user }) {
  return <p>{user}</p>;
}
```

Here, `Parent` doesn’t use `user`, but it still passes it to `Child` → that’s **prop drilling**.

---

### ⚡ **Why It’s a Problem:**

* Leads to **tightly coupled components**.
* Makes the code **hard to maintain**.
* Every intermediate component must know about props it doesn’t need.

---

### ✅ **Alternatives:**

* **React Context API** (for global or shared state).
* **State management libraries** like Redux, Zustand, or Recoil.

---

Would you like me to ask a **medium React question** about **Context vs Redux** next?


Perfect follow-up 👏

Here’s how you can use the **React Context API** to avoid **prop drilling**:

---

### ⚙️ **Step-by-Step Example:**

```jsx
import React, { createContext, useContext } from "react";

// 1️⃣ Create a Context
const UserContext = createContext();

// 2️⃣ Provide the context at a high level
function App() {
  const user = "Alice";
  return (
    <UserContext.Provider value={user}>
      <Parent />
    </UserContext.Provider>
  );
}

// 3️⃣ Consume the context anywhere in the tree
function Child() {
  const user = useContext(UserContext);
  return <p>{user}</p>;
}

function Parent() {
  return <Child />; // no need to pass 'user' as prop!
}
```

---

### 🧠 **Explanation:**

* `createContext()` creates a **context object**.
* `Provider` wraps components that need access to shared data.
* `useContext()` allows **any descendant** to directly access that data.

---

✅ **Benefits:**

* Eliminates **prop drilling** entirely.
* Makes components **cleaner and more reusable**.
* Works great for **theme**, **user auth**, or **global settings**.

---

Would you like me to show how to use **Context + useReducer** together (for global state management like Redux)?
