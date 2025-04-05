A Git branch gets **diverged** when both the local branch and the remote branch have **new commits** that the other doesn’t have. This creates two distinct lines of development that must be reconciled by merging or rebasing.

---

### **How Divergence Happens**

Let’s say you and a teammate are working on the same branch (`main`):

#### Initial shared state:
```
A---B  (origin/main, your local main)
```

#### You create new commits locally:
```
A---B---C---D  (your local main)
```

#### Meanwhile, your teammate pushes new commits:
```
A---B---E---F  (origin/main)
```

Now your branch has **diverged** from the remote:
- Your commits: `C`, `D`
- Remote commits: `E`, `F`

---

### **How to Detect Divergence**
When you try to do a `git pull`, Git will warn:
```
Your branch and 'origin/main' have diverged,
and have 2 and 2 different commits each, respectively.
```

---

### **Resolving Divergence**

#### Option 1: Merge (default behavior)
```bash
git pull
```
This will merge remote changes and create a merge commit.

#### Option 2: Rebase (cleaner history)
```bash
git pull --rebase
```
This replays your commits on top of the updated remote branch.

#### Option 3: Manually handle
If you want more control:
```bash
git fetch
git log HEAD..origin/main --oneline  # View incoming changes
git rebase origin/main               # Or git merge origin/main
```

---

### **How to Avoid Divergence**
- Always `git pull --rebase` before pushing
- Use feature branches and squash/rebase before merge
- Communicate in teams to reduce concurrent pushes to shared branches

Let me know if you want a diagram or live example of this!