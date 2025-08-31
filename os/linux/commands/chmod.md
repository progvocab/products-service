— `chmod` is one of the most important commands in Linux/Unix for **managing file permissions**. Let’s go step by step.

---

# 🔐 1. What is `chmod`?

* `chmod` = **change mode**
* Used to **set or modify permissions** of files and directories in Linux/Unix systems.
* Permissions control **who** can **read, write, or execute** a file.

---

# 📂 2. Types of Permissions

Each file/directory has 3 types of permissions:

* **r** → read (view file content, list directory)
* **w** → write (modify file, create/delete inside directory)
* **x** → execute (run file as program, enter directory)

---

# 👥 3. Who Gets Permissions?

There are **three categories of users**:

* **u** = user (the file owner)
* **g** = group (users in the file’s group)
* **o** = others (everyone else)
* **a** = all (u + g + o)

---

# 📝 4. Viewing Permissions

Use `ls -l`:

```bash
ls -l file.txt
```

Output:

```
-rw-r--r--  1 alice staff 1234 Aug 31 10:00 file.txt
```

Breakdown:

* `-` → type (`-` = file, `d` = directory, `l` = symlink)
* `rw-` → user (owner) has read+write
* `r--` → group has read-only
* `r--` → others have read-only

---

# ⚙️ 5. Using `chmod`

There are **two ways** to set permissions:

### (A) Symbolic Mode

* Add/remove permissions with `+`, `-`, `=`

Examples:

```bash
chmod u+x file.sh      # Add execute for user
chmod g-w file.txt     # Remove write for group
chmod o= file.txt      # Remove all permissions for others
chmod a+r file.txt     # Give everyone read permission
```

---

### (B) Numeric (Octal) Mode

Each permission = a number:

* `r = 4`, `w = 2`, `x = 1`

Add them up:

* `7 = rwx`
* `6 = rw-`
* `5 = r-x`
* `4 = r--`
* `0 = ---`

Format: `chmod XYZ file`

* `X = user`, `Y = group`, `Z = others`

Examples:

```bash
chmod 755 script.sh   # rwx for user, r-x for group, r-x for others
chmod 644 file.txt    # rw- for user, r-- for group, r-- for others
chmod 700 private.txt # rwx for user only
```

---

# 🔑 6. Special Permissions

Beyond basic `rwx`, there are special bits:

1. **Setuid (4xxx)** → run program with owner’s privileges

   ```bash
   chmod 4755 program
   ```
2. **Setgid (2xxx)** → run with group’s privileges, or new files inherit group in directories

   ```bash
   chmod 2755 dir
   ```
3. **Sticky bit (1xxx)** → in directories, only owner can delete own files (common in `/tmp`)

   ```bash
   chmod 1777 /tmp
   ```

---

# 📊 7. Examples

```bash
chmod 644 report.txt   # User rw-, Group r--, Others r--
chmod 755 script.sh    # User rwx, Group r-x, Others r-x
chmod u+x script.sh    # Give execute to user
chmod g+w shared.txt   # Give write to group
chmod o-r secret.txt   # Remove read from others
```

---

# ✅ Summary

* **`chmod`** = change file permissions.
* **Permissions** = read (4), write (2), execute (1).
* **Users** = user (u), group (g), others (o), all (a).
* **Two ways**: symbolic (`u+x`) or numeric (`755`).
* **Special bits**: setuid, setgid, sticky bit.

---

👉 Would you like me to also create a **visual table/cheat sheet** (mapping symbolic vs numeric chmod) so you can memorize quickly?
