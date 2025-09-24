Good question 👍

When you’re in the middle of a **`git rebase` conflict**, and you want to **accept “their” change** (the commit being rebased on top of your branch), you can use:

```bash
git checkout --theirs <file>
```

or in modern Git (preferred):

```bash
git restore --source=THEIRS <file>
```

Then stage it:

```bash
git add <file>
```

and continue:

```bash
git rebase --continue
```

---

### 🧭 Quick recap of conflict sides:

* **OURS** → your current branch (the one you’re rebasing onto).
* **THEIRS** → the commit(s) being replayed during the rebase.

So in **rebase**,

* `--ours` = branch you’re rebasing onto
* `--theirs` = the commit being replayed

---

👉 Do you want me to also show the **equivalent command if you want to accept *your* changes** (keep what’s already on the branch you’re rebasing onto)?
