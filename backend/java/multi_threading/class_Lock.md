Let’s go step by step into **class lock vs object lock in Java multithreading**.

---

## 🔒 1. Object Lock (Instance Lock)

* Every **Java object** has an **intrinsic monitor lock**.
* When you declare a **synchronized instance method** or use a `synchronized (this)` block, the **object’s lock** is acquired.
* Only **one thread per object instance** can hold the lock at a time.
* Different objects of the same class have different locks.

### Example: Object Lock

```java
class Printer {
    public synchronized void printDocs(String name) {
        for (int i = 1; i <= 3; i++) {
            System.out.println(name + " printing page " + i);
        }
    }
}

public class ObjectLockExample {
    public static void main(String[] args) {
        Printer printer1 = new Printer();
        Printer printer2 = new Printer();

        // Thread-1 uses printer1
        Thread t1 = new Thread(() -> printer1.printDocs("Thread-1"));
        // Thread-2 also uses printer1 -> will block if t1 is printing
        Thread t2 = new Thread(() -> printer1.printDocs("Thread-2"));
        // Thread-3 uses printer2 -> no blocking with printer1
        Thread t3 = new Thread(() -> printer2.printDocs("Thread-3"));

        t1.start();
        t2.start();
        t3.start();
    }
}
```

👉 Here:

* `t1` and `t2` share the same object (`printer1`) → they **synchronize** on the same object lock.
* `t3` uses a different object (`printer2`) → runs **independently**.

---

## 🔒 2. Class Lock (Static Lock)

* Every **Java class** has a **Class object** (in metaspace).
* A `synchronized static` method uses the **class lock**, i.e., lock on the `Class` object.
* Ensures only one thread can execute **any static synchronized method** of that class at a time.
* The lock is **shared across all instances** of that class.

### Example: Class Lock

```java
class Database {
    public static synchronized void updateRecord(String threadName) {
        for (int i = 1; i <= 3; i++) {
            System.out.println(threadName + " updating record " + i);
        }
    }
}

public class ClassLockExample {
    public static void main(String[] args) {
        // Different threads, different objects, but static lock is shared
        Database db1 = new Database();
        Database db2 = new Database();

        Thread t1 = new Thread(() -> db1.updateRecord("Thread-1"));
        Thread t2 = new Thread(() -> db2.updateRecord("Thread-2"));

        t1.start();
        t2.start();
    }
}
```

👉 Here:

* Even though `db1` and `db2` are **different objects**, the **static synchronized method** locks on `Database.class`.
* So `t1` and `t2` cannot run `updateRecord()` in parallel.

---

## 📊 Comparison Table

| Feature          | Object Lock (Instance Lock)                    | Class Lock (Static Lock)                                     |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| Scope            | Per object instance                            | Per class (Class object)                                     |
| Used by          | `synchronized` instance methods / blocks       | `synchronized` static methods / blocks                       |
| Synchronization  | Only affects threads using the **same object** | Affects all threads using the class (regardless of instance) |
| Example Use Case | Printing per printer (per object lock)         | Updating a shared DB schema (class-level lock)               |

---

✅ **Rule of Thumb:**

* Use **object locks** when synchronizing access to instance-specific data.
* Use **class locks** when synchronizing access to static data shared across all objects.

---

Would you like me to also **draw a diagram** (showing thread interactions with class lock vs object lock) to make it crystal clear?


Excellent question 👍 — this is a common confusion when learning Java synchronization.
Let’s clear it step by step.

---

## 🔒 What is an **Object Lock**?

* In Java, every **object** has an **intrinsic lock** (a monitor lock).
* When a thread enters a `synchronized` **instance method** (or a `synchronized (this)` block), it must **acquire the lock of that object**.
* Only **one thread at a time** can hold that object’s lock.
* Other threads trying to enter **any synchronized instance method** of the same object will be **blocked**, even if they’re different methods.

---

## 🧩 Scenario: Two Different Synchronized Methods in One Object

```java
class Shared {
    public synchronized void methodA() {
        System.out.println(Thread.currentThread().getName() + " entered methodA");
        try { Thread.sleep(2000); } catch (InterruptedException e) {}
        System.out.println(Thread.currentThread().getName() + " exiting methodA");
    }

    public synchronized void methodB() {
        System.out.println(Thread.currentThread().getName() + " entered methodB");
        try { Thread.sleep(2000); } catch (InterruptedException e) {}
        System.out.println(Thread.currentThread().getName() + " exiting methodB");
    }
}

public class ObjectLockExample {
    public static void main(String[] args) {
        Shared shared = new Shared();

        Thread t1 = new Thread(shared::methodA, "Thread-1");
        Thread t2 = new Thread(shared::methodB, "Thread-2");

        t1.start();
        t2.start();
    }
}
```

### 🔎 What happens?

* `Thread-1` calls `methodA()` → acquires **object lock** on `shared`.
* `Thread-2` calls `methodB()` → also needs the **same object lock**.
* Since `Thread-1` holds the lock, `Thread-2` must **wait** until `Thread-1` exits.

👉 Even though `methodA` and `methodB` are different methods, they both synchronize on the **same object lock (`shared`)**, so they cannot run concurrently.

---

## ✅ Variations

1. **Different Objects → Different Locks**
   If two threads call `methodA()` and `methodB()` on **different objects**, they can run in parallel (since each object has its own lock).

   ```java
   Shared obj1 = new Shared();
   Shared obj2 = new Shared();

   new Thread(obj1::methodA).start();
   new Thread(obj2::methodB).start(); // runs in parallel
   ```

2. **Synchronized Instance vs Static Method**

   * Instance methods lock on the **object instance** (`this`).
   * Static synchronized methods lock on the **Class object** (`ClassName.class`).
   * They don’t block each other unless both are static.

---

## 📌 Final Answer

If a class has **two different synchronized instance methods**, then **two threads calling them on the same object will block each other**, because both methods require the **same object lock**.
They can only run in parallel if they’re called on **different instances** (different object locks).

---

👉 Would you like me to also draw a **diagram of threads, object locks, and method calls** to visualize this better?
