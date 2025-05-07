 Here's a **step-by-step guide** to set up **Buildkite CI/CD with GitHub**, including pipeline configuration and self-hosted agent setup.

---

## 🚀 Step-by-Step: Integrate Buildkite with GitHub

---

### ✅ 1. **Create a Buildkite Account & Organization**

* Go to [https://buildkite.com](https://buildkite.com)
* Create an account and a new **organization**
* Connect your GitHub account when prompted

---

### ✅ 2. **Create a New Pipeline**

1. Click on **"New Pipeline"**
2. Name your pipeline (e.g., `my-github-ci`)
3. Connect it to your GitHub repository (private or public)
4. Choose **"YAML steps"**
5. Buildkite will now look for a file: `.buildkite/pipeline.yml`

---

### ✅ 3. **Add `.buildkite/pipeline.yml` to Your GitHub Repo**

```yaml
# .buildkite/pipeline.yml
steps:
  - label: ":hammer: Build & Test"
    command:
      - ./gradlew build
    agents:
      queue: "default"
```

> You can customize this with multiple steps, parallelism, conditionals, etc.

---

### ✅ 4. **Install & Start a Buildkite Agent (Self-Hosted)**

#### Install agent on a Linux/Mac/Windows machine:

```bash
# Install Buildkite Agent
sudo bash -c "$(curl -L https://buildkite.com/install.sh)"

# Configure the agent with your token (from Buildkite UI)
buildkite-agent start
```

> You'll find the **agent token** in the pipeline settings under **Agents**.

#### Agent runs:

* `buildkite-agent start` launches and listens for jobs.
* Agent executes jobs locally, like GitHub Actions runner.

---

### ✅ 5. **Push Code to Trigger the Build**

* Push a commit to GitHub
* GitHub sends a webhook to Buildkite
* Buildkite triggers the pipeline
* Self-hosted agent picks the job, executes it
* Build status is sent back to the GitHub PR/commit

---

## 🧠 Bonus: Example Pipeline with Multiple Steps

```yaml
steps:
  - label: ":white_check_mark: Lint"
    command: ./gradlew check

  - label: ":test_tube: Unit Tests"
    command: ./gradlew test
    parallelism: 2

  - wait

  - label: ":package: Build"
    command: ./gradlew assemble

  - label: ":rocket: Deploy"
    command: ./scripts/deploy.sh
    if: build.branch == "main"
```

---

### 🔐 Optional: Set Secrets and Environment Variables

In Buildkite UI:

* Go to **Pipeline > Settings > Environment Hooks**
* Set sensitive keys like `DATABASE_URL`, `API_TOKEN`, etc.

---

## 🔁 Webhook Setup (Optional Manual)

If not auto-configured, you can add a webhook manually in GitHub:

1. Go to **GitHub > Settings > Webhooks**
2. URL: `https://webhook.buildkite.com/deliver/ORG-TOKEN`
3. Content type: `application/json`

---

Would you like an example for a **Java or Spring Boot CI/CD pipeline using Buildkite**?
