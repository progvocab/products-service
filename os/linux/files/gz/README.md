Perfect question 👍. Log files are often rotated and compressed into **`.gz`** format (gzip). You don’t need to decompress them manually — Linux provides tools to **view, search, and read `.gz` files directly**.

---

## 🔹 Viewing `.gz` Files

### 1. **zcat** → dump contents to terminal

```bash
zcat file.log.gz
```

* Similar to `cat` but for gzip files.
* Streams the whole file (can be large).

---

### 2. **zless** → scrollable view (like `less`)

```bash
zless file.log.gz
```

* Lets you page through logs interactively.
* Use `↑ ↓ PgUp PgDn / ? n N` for navigation and search.

---

### 3. **zmore** → similar to `more`

```bash
zmore file.log.gz
```

---

### 4. **zgrep** → search inside `.gz` without extracting

```bash
zgrep "ERROR" file.log.gz
```

* Works just like `grep` on compressed files.
* Add options like `-i` (case-insensitive), `-n` (line numbers), `-H` (filenames).

Example:

```bash
zgrep -in "exception" /var/log/*.gz
```

---

### 5. **gunzip -c** → decompress to stdout (without extracting)

```bash
gunzip -c file.log.gz | less
```

---

### 6. **gzcat** (on some systems like macOS, BSD)

```bash
gzcat file.log.gz
```

(Same as `zcat`.)

---

✅ **Summary**

* Quick view → `zcat file.log.gz`
* Scrollable → `zless file.log.gz`
* Search → `zgrep "ERROR" file.log.gz`

---

👉 Do you want me to also show you how to **search for “ERROR” across multiple `.gz` log files in a directory at once**?
