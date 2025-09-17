

In **Java Garbage Collection (GC)**, two important terms often come up: **mutator** and **collector**.

---

## 1. Mutator

* The **mutator** is **your Java application code** — the part of the program that **allocates objects, updates references, and mutates the heap**.
* Every time you do `new Object()` or update a field/reference, you are acting as the **mutator**.

👉 In short: **Mutator = Application code that creates and modifies objects in the heap**.

---

## 2. Collector

* The **collector** is the **garbage collector thread/system** that **reclaims memory** by finding objects that are no longer reachable and freeing that memory.
* It runs **alongside or in pauses of the mutator**.
* Collector ensures you don’t run out of heap space.

👉 In short: **Collector = JVM’s Garbage Collector that reclaims unreachable objects**.

---

## 3. Mutator and Collector Interaction

* The **mutator keeps creating objects**.
* When heap space becomes tight, the **collector runs** (sometimes stopping the mutator briefly, sometimes concurrently depending on GC algorithm like G1, ZGC, Shenandoah).
* After GC, the mutator resumes execution.

This balance allows Java developers to focus on logic instead of manual memory management.

---

## 4. Code Example – Mutator

```java
public class MutatorExample {
    public static void main(String[] args) {
        // Mutator: allocating objects
        for (int i = 0; i < 100000; i++) {
            String data = new String("Object " + i); // creates objects in heap
        }

        System.out.println("Objects created. Waiting for collector...");
        
        // Hinting garbage collection
        System.gc();  // Suggest JVM to run the collector (not guaranteed)
    }
}
```

Here:

* The **mutator** is the loop allocating many `String` objects.
* These objects eventually become unreachable.
* The **collector** will reclaim them when GC runs.

---

## 5. Code Example – Showing Collector Activity

We can monitor collector activity using **`-verbose:gc`** or newer **JVM options**:

```bash
java -verbose:gc MutatorExample
```

Or more detailed logging (Java 9+):

```bash
java -Xlog:gc MutatorExample
```

This shows when the **collector** runs and how much memory was reclaimed.

---

## 6. Example with Weak References (Collector reclaiming)

```java
import java.lang.ref.WeakReference;

public class CollectorExample {
    public static void main(String[] args) {
        Object strong = new Object();           // Mutator creates object
        WeakReference<Object> weak = new WeakReference<>(strong);

        strong = null;  // Remove strong reference

        System.gc();    // Suggest GC to run

        if (weak.get() == null) {
            System.out.println("Collector has reclaimed the object!");
        } else {
            System.out.println("Object still alive!");
        }
    }
}
```

Here:

* The **mutator** creates an object and drops the strong reference.
* The **collector** reclaims it since only a weak reference is left.

---

✅ **Summary:**

* **Mutator** = Your program that creates/updates objects.
* **Collector** = JVM GC that reclaims unused objects.
* They work together: mutator grows the heap, collector keeps it from overflowing.

---

Do you want me to also **draw a diagram of mutator and collector interaction in Java GC** (like how they alternate between execution and pause)?
