A **flaky test** is a test that **sometimes passes and sometimes fails**, even though the code under test has not changed and the underlying functionality is correct.

In other words, flaky tests are **non-deterministic** — their outcome varies between runs, making them unreliable indicators of code quality.

---

### Causes of Flaky Tests

Some common reasons why tests become flaky:

* **Timing issues / race conditions**
  e.g., asynchronous code, multithreading, waiting for resources.

* **Test order dependency**
  The result of one test depends on whether another test ran before it.

* **External dependencies**
  Tests that rely on databases, APIs, networks, or file systems may behave differently depending on system state.

* **Environment issues**
  Different OS, hardware speed, or resource availability cause inconsistencies.

* **Randomness in tests**
  If tests use random data without fixed seeds, outcomes may differ.

---

### Why Flaky Tests Are Problematic

* They reduce **trust in the test suite**.
* They increase **time and cost of debugging**.
* They may **hide real bugs** or **block deployments** (CI/CD pipelines).

---

### Example

Suppose you write a test for an async function:

```java
@Test
public void testAsyncService() {
    String result = asyncService.call();
    assertEquals("SUCCESS", result); // Sometimes passes, sometimes fails
}
```

If the test doesn’t wait properly for the async call to complete, it might check too early, making the test flaky.

---

✅ **In short:**
A flaky test is one that gives inconsistent results due to issues in the test code, test environment, or dependencies — not necessarily because the application itself is broken.

---

Do you want me to also explain **how to detect and fix flaky tests** in real projects (like CI/CD pipelines)?
