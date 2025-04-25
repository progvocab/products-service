The `pyproject.toml` file is the modern standard configuration file for Python projects. It defines how tools (like `poetry`, `setuptools`, `black`, etc.) should build and manage the project. It replaces older config files like `setup.py`, `setup.cfg`, and `requirements.txt` in many cases.

---

### **Basic Structure of `pyproject.toml`**

```toml
[project]
name = "my_project"
version = "0.1.0"
description = "A simple example project"
authors = [
  { name="Alice", email="alice@example.com" }
]
dependencies = [
  "requests>=2.25.1",
  "pydantic"
]
requires-python = ">=3.8"

[project.optional-dependencies]
dev = ["pytest", "black"]
docs = ["mkdocs"]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
# Specific config for setuptools

[tool.poetry]
# Poetry-specific config (alternative to [project] when using Poetry)
```

---

### **Common Sections Explained**

#### **[project]**
Standard PEP 621 metadata section for project info and dependencies.

| Key              | Description                                |
|------------------|--------------------------------------------|
| `name`           | Name of your package                       |
| `version`        | Version string                             |
| `description`    | Short description                          |
| `authors`        | List of authors                            |
| `dependencies`   | Runtime dependencies                       |
| `optional-dependencies` | Grouped optional packages          |
| `requires-python`| Minimum Python version                     |

---

#### **[build-system]**
Tells Python how to build your project.

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

If you use Poetry:

```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

---

#### **[tool.<toolname>]**
Configuration for tools like `black`, `isort`, `pytest`, `mypy`, etc.

Example for `black`:
```toml
[tool.black]
line-length = 88
target-version = ["py38"]
```

Example for `mypy`:
```toml
[tool.mypy]
strict = true
```

---

### **Using `pyproject.toml` with Tools**

| Tool        | Usage Section             |
|-------------|---------------------------|
| `poetry`    | `[tool.poetry]`           |
| `setuptools`| `[project]`, `[build-system]` |
| `black`     | `[tool.black]`            |
| `pytest`    | `[tool.pytest.ini_options]` |
| `ruff`, `mypy` | `[tool.ruff]`, `[tool.mypy]`|

---

Would you like a sample complete `pyproject.toml` for a real-world project using Poetry, Black, and Mypy?