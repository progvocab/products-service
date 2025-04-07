Contributing to **ClickHouse** is a great way to dive into high-performance database internals and build your C++ systems skills. Here's a comprehensive guide to help you get started:

---

## **1. Fork & Clone the Repository**

**GitHub Repo**: [https://github.com/ClickHouse/ClickHouse](https://github.com/ClickHouse/ClickHouse)

```bash
git clone --recursive https://github.com/ClickHouse/ClickHouse.git
cd ClickHouse
```

Use `--recursive` to clone submodules.

---

## **2. Set Up Development Environment**

### **Dependencies (Ubuntu/Debian)**
Install build tools and libraries:

```bash
sudo apt update
sudo apt install -y \
  cmake ninja-build ccache clang \
  libssl-dev libicu-dev libunwind-dev \
  zlib1g-dev liblz4-dev libzstd-dev \
  libreadline-dev libpoco-dev \
  libboost-all-dev python3 python3-pip
```

Install CMake ≥ 3.23 if needed via [Kitware APT](https://apt.kitware.com/).

---

## **3. Build ClickHouse (Debug Mode)**

```bash
mkdir build
cd build
cmake .. -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug -GNinja
ninja clickhouse
```

This builds the ClickHouse server binary with tests.

---

## **4. Run ClickHouse Server Locally**

After building:

```bash
./programs/clickhouse server
```

Or if you built using CMake in `build`:

```bash
./build/programs/clickhouse server
```

Default config and logs are under:
- Config: `programs/server/config.xml`
- Data: `./data/`
- Logs: `./logs/`

You can open a client shell in another terminal:

```bash
./programs/clickhouse client
```

---

## **5. Run Tests**

You can run tests with:

```bash
./tests/run.py
```

Or specific integration tests:

```bash
./tests/clickhouse-test --print-time --shard 1 --jobs 4
```

---

## **6. Good First Issues**

Check out:  
[https://github.com/ClickHouse/ClickHouse/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22](https://github.com/ClickHouse/ClickHouse/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

---

## **7. Coding Standards**

- ClickHouse is written in **modern C++ (C++20)**.
- Format code with `clang-format`.
- Follow [ClickHouse Style Guide](https://github.com/ClickHouse/ClickHouse/blob/master/CONTRIBUTING.md#style-guide).

---

## **8. Submit a PR**

1. Make changes in a feature branch.
2. Push and open a pull request.
3. Include tests and benchmark results if applicable.

---

## **9. Join the Community**

- [Telegram Chat](https://t.me/clickhouse_en)
- [GitHub Discussions](https://github.com/ClickHouse/ClickHouse/discussions)
- [Slack (invite-only)](https://clickhouse.com/slack)
- [Weekly Meetups](https://www.youtube.com/c/AltinityInc)

---

Would you like a Docker-based or IDE-integrated dev environment setup too (e.g., VS Code + Remote Containers)?