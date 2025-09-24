When you run `git rebase` and Git finds conflicts, the rebase **pauses** and asks you to resolve them before continuing. Here’s the proper flow:

---

### ✅ Steps after conflicts in `git rebase`

1. **Check which files have conflicts**

   ```bash
   git status
   ```

   You’ll see a list of files marked as **unmerged**.

2. **Open and fix conflicts in those files**
   Conflicted areas are marked like this:

   ```diff
   <<<<<<< HEAD
   your branch’s changes
   =======
   the changes from the commit being rebased
   >>>>>>> commit-id
   ```

   * Edit the file(s) to keep the correct code.
   * Remove the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).

3. **Mark the conflicts as resolved**
   After fixing each conflicted file:

   ```bash
   git add <file>
   ```

   or to stage all resolved files:

   ```bash
   git add .
   ```

4. **Continue the rebase**

   ```bash
   git rebase --continue
   ```

   * If there are more commits to reapply, Git might show conflicts again — repeat steps 1–4.
   * If there are no more conflicts, the rebase finishes.

5. **If you need to stop the rebase**

   * To **abort** and return to the state before rebase:

     ```bash
     git rebase --abort
     ```
   * To **skip** the problematic commit (not recommended unless you’re sure):

     ```bash
     git rebase --skip
     ```

---

### 🔑 Summary of commands you’ll use the most:

* `git status` → check conflicts
* fix files manually
* `git add <file>` → mark resolved
* `git rebase --continue` → proceed

---

Do you want me to also show you a **real-world example** with conflicting commits and how to resolve them step by step?
