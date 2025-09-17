The `uv sync` command is used in the **[uv package manager](https://github.com/astral-sh/uv)** — a **fast, Python-compatible package manager and virtual environment tool** developed as a Rust-powered alternative to pip and venv.

---

### **What `uv sync` Does**

`uv sync` installs all dependencies listed in your `pyproject.toml` and `poetry.lock` or `requirements.txt`, **syncing your environment** to match the locked versions.

It ensures that your environment is **consistent** and **reproducible**.

---

### **Usage**

```bash
uv sync
```

This will:

- Read from `pyproject.toml` and `poetry.lock` (if using Poetry)
- Or `requirements.txt` (if present)
- Install all dependencies
- Create or use the virtual environment under `.venv` by default
- Pin the versions exactly to avoid drift

---

### **Example (with pyproject.toml + poetry.lock)**

```toml
# pyproject.toml
[tool.poetry.dependencies]
requests = "^2.31.0"
```

```bash
uv sync
```

This ensures that the version of `requests` is locked and installed.

---

### **Benefits Over pip**

- Much faster (written in Rust)
- Automatic virtual environment handling
- Lockfile consistency (like Poetry)
- Optional editable installs
- Compatible with existing ecosystem (pip, requirements.txt)

---

### **Options**

```bash
uv sync --dev         # Include dev dependencies
uv sync --no-install  # Only resolves and writes lockfile
uv sync --python 3.11 # Specify Python version
```

---

Would you like a comparison between `uv`, `pip`, and `poetry` for managing environments and dependencies?

Here’s a **comparison of `uv`, `pip`, and `poetry`** for Python dependency and environment management:

---

### **1. Overview**

| Feature              | `uv`                                  | `pip`                             | `poetry`                             |
|----------------------|----------------------------------------|------------------------------------|---------------------------------------|
| Language             | Rust (ultra-fast)                      | Python (default)                   | Python                                |
| Speed                | Extremely fast                         | Moderate                           | Slower than `uv`, faster than `pip` sometimes |
| Lockfile Support     | Yes (compatible with `poetry.lock`)    | No (relies on `requirements.txt`) | Yes (`poetry.lock`)                  |
| Virtual Env Mgmt     | Automatic `.venv`                      | Needs `venv` or `virtualenv`       | Automatic `.venv`                    |
| Dependency Resolver  | Modern and strict                      | Basic                              | Modern and strict                    |
| Editable Installs    | Yes                                    | Yes                                | Partial                              |
| Dev Dependency Split | Yes (`--dev`)                          | No (manual)                        | Yes                                   |

---

### **2. Installation & Setup**

#### `uv`
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv
uv sync
```

#### `pip`
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### `poetry`
```bash
pip install poetry
poetry install
```

---

### **3. Managing Dependencies**

| Task                    | `uv`                 | `pip`                         | `poetry`              |
|-------------------------|----------------------|--------------------------------|------------------------|
| Add Dependency          | `uv pip install foo` | `pip install foo`             | `poetry add foo`       |
| Install All             | `uv sync`            | `pip install -r requirements.txt` | `poetry install`     |
| Remove Dependency       | Manual               | Manual                         | `poetry remove foo`    |
| Lock Dependencies       | `poetry.lock`        | None (manual freeze)           | Auto lock               |

---

### **4. Speed**

| Operation               | `uv`       | `pip`     | `poetry`  |
|-------------------------|------------|-----------|------------|
| Environment creation    | Fast       | Moderate  | Moderate   |
| Package install         | Super fast| Slower    | Slower     |

---

### **5. Best Use Cases**

| Tool    | Use Case                                                                 |
|---------|--------------------------------------------------------------------------|
| `uv`    | Fast dependency sync, modern workflows, Poetry-compatible                |
| `pip`   | Lightweight, traditional workflows, custom tooling needs                 |
| `poetry`| Full project management, publishing to PyPI, clean CLI experience         |

---

Would you like a sample project setup using `uv` and `poetry` side by side to demonstrate compatibility?