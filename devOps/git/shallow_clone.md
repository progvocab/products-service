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
