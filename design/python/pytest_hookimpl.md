### **Understanding `@hookimpl` with `tryfirst`, `trylast`, and `purest` in Pluggy**
Pluggy is a **plugin management framework** used by tools like **pytest**, **tox**, and **pre-commit**. The `@hookimpl` decorator in Pluggy is used to **register hook implementations**.

---

## **1️⃣ What is `@hookimpl` in Pluggy?**
In Pluggy, plugins define hooks using `@hookimpl`. You can **control the order of hook execution** using:
- `tryfirst=True` → Runs **before** other hooks.
- `trylast=True` → Runs **after** other hooks.
- `purest=True` → Ensures **only one** hook implementation exists.

---

## **2️⃣ What is `purest=True` in `@hookimpl`?**
The `purest=True` argument **raises an error if multiple implementations exist** for the same hook.  
This ensures that **only one** plugin provides a specific implementation.

👉 **Use case:**  
If you want to guarantee that **no conflicting implementations** of a hook exist.

---

## **3️⃣ Example: Using `purest=True`**
Let’s define a **hook specification** and multiple implementations to see how `purest=True` behaves.

### **🔹 Step 1: Define Hook Specification (Hook Definition)**
First, create a **hook specification** that plugins must implement.

```python
import pluggy

hookspec = pluggy.HookspecMarker("myplugin")
hookimpl = pluggy.HookimplMarker("myplugin")

class MySpec:
    """Define a hook specification."""
    @hookspec
    def process_data(self, data):
        """Processes data and returns the result."""
        pass
```

---

### **🔹 Step 2: Define Multiple Implementations Without `purest`**
Here, we define **two plugins** implementing `process_data()`.

```python
class Plugin1:
    @hookimpl
    def process_data(self, data):
        return data.upper()

class Plugin2:
    @hookimpl
    def process_data(self, data):
        return data[::-1]  # Reverses the string
```
👉 **Problem:** Since multiple implementations exist, Pluggy will call **both** and return a list of results.

---

### **🔹 Step 3: Enforce a Single Implementation Using `purest=True`**
Modify one of the implementations to enforce exclusivity.

```python
class Plugin1:
    @hookimpl(purest=True)
    def process_data(self, data):
        return data.upper()
```

Now, if another plugin also implements `process_data()`, Pluggy **throws an error**.

---

### **4️⃣ Example: Running Pluggy with `purest=True`**
Let’s register the plugins and test what happens.

```python
pm = pluggy.PluginManager("myplugin")
pm.add_hookspecs(MySpec)

plugin1 = Plugin1()
plugin2 = Plugin2()

pm.register(plugin1)
pm.register(plugin2)  # This will cause an error because Plugin1 has `purest=True`

result = pm.hook.process_data(data="hello")
print(result)  # Expected: ERROR, since Plugin1 declared `purest=True`
```

---

## **5️⃣ What Happens When `purest=True` is Used?**
1. If multiple plugins implement the same hook:
   - **Without `purest=True`** → Pluggy calls all implementations and returns a list.
   - **With `purest=True`** → Pluggy **raises an error** if another plugin tries to implement it.

2. **When to use `purest=True`?**
   - When you want **one and only one** plugin to handle a hook.
   - Prevents conflicts when multiple plugins try to control the same functionality.

---

## **6️⃣ Summary Table**
| Parameter | Effect |
|-----------|--------|
| `@hookimpl(tryfirst=True)` | Runs **before** other implementations. |
| `@hookimpl(trylast=True)` | Runs **after** other implementations. |
| `@hookimpl(purest=True)` | Ensures **only one** implementation exists. |

Would you like an example with **pytest plugins** using `purest=True`? 🚀
