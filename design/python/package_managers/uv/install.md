Got it — you're using [`uv`](https://github.com/astral-sh/uv), the new ultra-fast Python package manager.

To download and install dependencies defined in `pyproject.toml` using **`uv`**, here's what you do:

---

### **1. Install all dependencies**
To install the project dependencies (like `poetry install`):

```bash
uv pip install -r pyproject.toml
```

But more commonly, you use `uv` with a **lock file** (`requirements.txt` or `requirements.lock`). If you're working with `pyproject.toml`, you can use `uv` with `uv venv`:

```bash
uv venv install
```

This:
- Creates a virtual environment
- Installs dependencies defined in `pyproject.toml`

---

### **2. Install with optional extras**

If your `pyproject.toml` defines optional extras, you can install like this:

```bash
uv pip install .[dev]
```

Or for multiple groups:

```bash
uv pip install .[dev,test]
```

---

### **3. Add a new dependency**

```bash
uv pip install requests
```

This installs the package and updates the `pyproject.toml` and lock file if managed.

---

### **4. Lock dependencies**

```bash
uv pip compile pyproject.toml --output requirements.txt
```

This resolves and locks all transitive dependencies.

---

Let me know if you're using `uv` with `hatch`, `setuptools`, or just plain `uv` so I can tailor it better.