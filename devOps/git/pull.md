Great question! Understanding how `git pull` behaves with **rebase**, **merge**, and **fast-forward** is crucial for maintaining a clean Git history.

---

### **1. `git pull` Recap**
`git pull` = `git fetch` + `git merge` (by default)  
But you can customize its behavior using:
- `--rebase=true`
- `--rebase=false`
- `--ff-only` (fast-forward only)

---

### **2. `--rebase=true` (or `git pull --rebase`)**
- Instead of merging your local commits with fetched ones, **your local changes are temporarily removed**, the remote changes are applied, then **your changes are replayed on top** (rebased).
- This gives a **linear, cleaner history**.

**Example:**
```bash
git pull --rebase
```

**Before:**
```
A---B (origin/main)
     \
      C---D (main)
```

**After rebase:**
```
A---B---C'---D' (main)
```

**Pros:**
- Clean, linear history
- Avoids unnecessary merge commits

**Cons:**
- Rewriting history (so avoid on shared branches unless you're careful)

---

### **3. `--rebase=false` (default behavior)**
- This is the **default behavior** if not set otherwise.
- `git pull` will perform a **merge**, creating a **merge commit**.

**Example:**
```bash
git pull --no-rebase
```

**Result:**
```
A---B---M (main)
     \ /
      C---D
```
- `M` is the **merge commit**.

**Pros:**
- Easier to trace how branches were merged
- Good for teamwork visibility

**Cons:**
- Messier history with extra merge commits

---

### **4. `--ff-only` (fast-forward only)**
- Git will **only update** your branch if it can do a **fast-forward merge**, i.e., **no new local commits**.

**Example:**
```bash
git pull --ff-only
```

- If there's divergence, Git will **refuse to merge** and ask you to handle it explicitly.

**Pros:**
- Ensures linear history
- Avoids automatic merges

**Cons:**
- Requires local branch to be strictly behind remote

---

### Summary Table

| Option           | Behavior                      | History         | Use Case                              |
|------------------|-------------------------------|------------------|----------------------------------------|
| `--rebase=true`  | Rebase local on remote        | Linear           | Clean history, personal branches       |
| `--rebase=false` | Merge remote into local       | Merge commits    | Default, collaborative workflows       |
| `--ff-only`      | Only fast-forward if possible | Linear only      | CI/CD, production branches             |

---

Let me know if you want a visual Git example or to simulate this with a repo!