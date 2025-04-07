Contributing a minor change to ClickHouse is straightforward once you set up the local development environment. Here's a step-by-step guide to make a change, build, and test it:

---

### **1. Fork and Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/ClickHouse.git
cd ClickHouse
git remote add upstream https://github.com/ClickHouse/ClickHouse.git
```

---

### **2. Install Dependencies**

ClickHouse requires **CMake**, **Ninja**, **GCC or Clang**, and some development libraries. On Ubuntu:

```bash
sudo apt update
sudo apt install -y git cmake ninja-build clang \
  libssl-dev libicu-dev libzstd-dev liblz4-dev \
  libprotobuf-dev protobuf-compiler libunwind-dev \
  libcurl4-openssl-dev libmysqlclient-dev libpq-dev
```

More details in the [official build instructions](https://github.com/ClickHouse/ClickHouse/blob/master/README.md#how-to-build-clickhouse).

---

### **3. Create a Build Directory and Configure**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
```

For debug builds:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -GNinja
```

---

### **4. Make a Minor Code Change**

For example, modify a message in `src/Interpreters/InterpreterSelectQuery.cpp` or a SQL function’s behavior.

> Use `grep` or `rg` (ripgrep) to search the codebase:
```bash
rg "SELECT"
```

---

### **5. Build ClickHouse**

```bash
ninja clickhouse
```

This will produce a binary: `./programs/clickhouse`

---

### **6. Run the Server Locally**

```bash
./programs/clickhouse server
```

Default ports:
- HTTP: `8123`
- Native client: `9000`
- TCP inter-server: `9010`

---

### **7. Run Tests**

ClickHouse has unit and integration tests using the `clickhouse-test` script.

Run a basic test set:
```bash
./programs/clickhouse-test
```

Run a specific test:
```bash
./programs/clickhouse-test --testname your_test.sql
```

Or use CTest:
```bash
cd ../build
ctest -R MergeTree
```

---

### **8. Commit and Push Changes**

```bash
git checkout -b my-fix
git add <file>
git commit -m "Fix: my minor change description"
git push origin my-fix
```

---

### **9. Create a Pull Request**

Open GitHub, navigate to your fork, and create a PR against `ClickHouse/ClickHouse:master`.

---

### Optional Tips:

- Use `ccache` to speed up rebuilds.
- Explore integration tests under `tests/queries` and functional test framework in `tests/`.
- Read [CONTRIBUTING.md](https://github.com/ClickHouse/ClickHouse/blob/master/CONTRIBUTING.md).

---

Let me know what kind of change you're thinking about — I can guide you directly to the file or logic path.