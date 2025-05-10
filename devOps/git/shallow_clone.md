Using `--depth=1` while cloning a Git repository performs a **shallow clone**, meaning it **only fetches the latest commit** on each branch instead of the full commit history.

---

### **Command Example**

```bash
git clone --depth=1 https://github.com/user/repo.git
```

---

### **What It Does**

* Downloads only the most recent commit.
* Skips the rest of the commit history (no ancestry, no older commits).
* Results in a **much faster and smaller** clone.
* Ideal for:

  * CI/CD pipelines
  * Limited storage environments
  * When you only need the latest code, not history

---

### **Limitations**

* You cannot view or checkout previous commits.
* Git commands that require history (e.g., `git log`, `git blame` across history) won’t work fully.
* Rebasing or merging may be restricted.

---

Would you like to know how to later convert a shallow clone into a full clone?


To convert a **shallow clone** (made with `--depth=1`) into a **full clone**, you need to **fetch the remaining commit history** from the remote.

---

### **Step-by-step to convert shallow to full clone**

1. **Navigate to your cloned repo**:

   ```bash
   cd repo
   ```

2. **Fetch the full history**:

   ```bash
   git fetch --unshallow
   ```

   This will fetch all previous commits and turn your shallow clone into a full clone.

   * Alternatively, if `--unshallow` is not supported (older versions):

     ```bash
     git fetch --depth=1000000
     ```

3. **Verify**:

   ```bash
   git log
   ```

   You should now see the complete commit history.

---

Would you like to automate this as part of a script or CI workflow?

