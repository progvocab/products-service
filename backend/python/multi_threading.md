 use **parallel processing** in Python to improve the performance of code—especially algorithms that are **CPU-bound** (e.g., graph traversal, number crunching, image processing).

---

### ✅ **When to Use Parallel Processing**

Use it when:

* Your task is **CPU-intensive** (e.g., algorithms, simulations, cryptography)
* You want to run **independent computations** simultaneously

Use **multithreading** for **I/O-bound** tasks (e.g., network calls, file reads).

---

## 🔧 Popular Options in Python

| Library              | Type                   | Best For                    |
| -------------------- | ---------------------- | --------------------------- |
| `multiprocessing`    | Parallel (multi-core)  | CPU-bound tasks             |
| `concurrent.futures` | Simplified parallelism | Both Thread & Process pools |
| `joblib`             | Simple parallel loops  | ML, data tasks              |
| `threading`          | Concurrency            | I/O-bound tasks             |
| `asyncio`            | Asynchronous           | High-throughput I/O         |

---

### 🧪 Example using `multiprocessing`

```python
from multiprocessing import Pool
import time

def square(n):
    return n * n

if __name__ == "__main__":
    numbers = list(range(10))
    with Pool(processes=4) as pool:
        results = pool.map(square, numbers)
    print(results)
```

⏱ This is often **much faster** than looping through `square(n)` sequentially, especially on multicore CPUs.

---

### ⚡ Example: Using `concurrent.futures.ProcessPoolExecutor`

```python
from concurrent.futures import ProcessPoolExecutor

def cube(n):
    return n ** 3

if __name__ == "__main__":
    nums = list(range(10))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(cube, nums))
    print(results)
```

---

### 🚀 Real-world Use Cases

* Graph algorithms (parallel BFS/DFS on subgraphs)
* Image processing (filtering multiple images at once)
* Matrix operations (parallel row/column processing)
* Data preprocessing for ML pipelines
* Web scraping (parallel URL fetch with threading/async)

---

### ⚠️ Caution:

* The **Global Interpreter Lock (GIL)** in CPython limits true threading for CPU-bound tasks—so use **multiprocessing** or **native C extensions**.
* Avoid parallelism for small or trivial tasks—it can add overhead.

---

Would you like an example that applies parallel processing to a specific algorithm (e.g., Tarjan’s, DFS, matrix multiplication)?
