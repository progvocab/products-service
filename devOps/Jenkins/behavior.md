### **What is "Behavior" in Jenkins Branch Sources?**  

In **Jenkins Multibranch Pipelines** and **Jenkins Organization Folders**, **"Behaviors"** define how Jenkins discovers, builds, and manages branches and pull requests from a source control system (like GitHub, GitLab, or Bitbucket).  

These behaviors help Jenkins determine:  
- **Which branches or pull requests to include/exclude.**  
- **How often to scan repositories for changes.**  
- **How to handle branch naming patterns.**  

---

## **1. Where to Find "Behaviors" in Jenkins?**  
In a **Multibranch Pipeline** or **Organization Folder**:  
1. **Go to your Jenkins job configuration.**  
2. Find the **"Branch Sources"** section.  
3. Click on **"Add"** or **"Add Behavior"** under the selected repository.  
4. Choose from the available behaviors.

---

## **2. Common "Behaviors" in Jenkins Branch Sources**
Each branch source plugin (e.g., Git, GitHub, Bitbucket) provides different behaviors. Here are some common ones:

### **a) Discover Branches**
- **What it does:** Allows Jenkins to scan and build branches.  
- **Options:**  
  - `Exclude branches that are also filed as PRs` (default)  
  - `Only branches that are also filed as PRs`  
  - `All branches`  

---

### **b) Discover Pull Requests**
- **What it does:** Defines how Jenkins handles pull requests.  
- **Options:**  
  - `Merging the pull request with the target branch` (default)  
  - `The head of the pull request (unmerged)`  
  - `Both merged and unmerged versions`  

---

### **c) Filter by Name (Regex)**
- **What it does:** Allows Jenkins to include or exclude branches based on **regular expressions (regex)**.  
- **Example:**  
  - To build only `feature/*` branches:  
    ```regex
    ^feature/.*
    ```
  - To exclude `hotfix/*` branches:  
    ```regex
    ^(?!hotfix/).*
    ```

---

### **d) Build Strategies (for GitHub, GitLab, etc.)**
- **What it does:** Controls when a branch or pull request is built.  
- **Common strategies:**  
  - Build branches with recent changes.  
  - Build PRs when there are new commits.  
  - Ignore old branches that haven’t changed for a while.

---

### **e) Periodic Scans of the Repository**
- **What it does:** Controls how often Jenkins scans for new branches or updates.  
- **Example:**  
  - Scan every **15 minutes**: `H/15 * * * *` (Cron syntax)

---

### **3. Example Use Case: Configuring Behaviors in a GitHub Multibranch Pipeline**
**Goal:**  
- Build only `feature/*` branches and pull requests that are **merged with the target branch**.

**Steps:**  
1. Add **GitHub Branch Source**.  
2. Under "Behaviors," select:
   - ✅ **Discover Branches** → `All branches`
   - ✅ **Discover Pull Requests** → `Merging the pull request with the target branch`
   - ✅ **Filter by Name (Regex)** → `^feature/.*`
3. Save and apply changes.

**Result:**  
- Jenkins will **only build** `feature/*` branches and merged pull requests.

---

## **4. Why Are "Behaviors" Useful?**
- **Reduces unnecessary builds** (by excluding irrelevant branches).  
- **Improves performance** (by scanning repositories efficiently).  
- **Enforces rules** for branch naming and build triggers.  
- **Automates CI/CD workflows** effectively.

---

## **Conclusion**
Jenkins **Branch Source Behaviors** control how branches and pull requests are discovered and built. Configuring them correctly can **optimize pipeline execution**, reduce **unnecessary builds**, and **align Jenkins with your development workflow**.

Would you like a specific example for your use case?