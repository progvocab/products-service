Here's a **comprehensive list of GitHub Actions events** and how to trigger actions using them in your workflow `.yml` files inside `.github/workflows/`.

---

### **1. Common Events & Triggers**

| **Event**             | **Description**                                                     |
| --------------------- | ------------------------------------------------------------------- |
| `push`                | Triggered on a push to the repository.                              |
| `pull_request`        | Triggered when a pull request is opened, reopened, or synchronized. |
| `workflow_dispatch`   | Manual trigger from GitHub UI.                                      |
| `schedule`            | Triggers based on a cron syntax (like CRON jobs).                   |
| `release`             | Triggered when a new release is published.                          |
| `issues`              | Triggered on issue activity (open, close, comment).                 |
| `issue_comment`       | Triggered when a comment is created on an issue or PR.              |
| `pull_request_review` | When a PR review is submitted.                                      |
| `create`              | Triggered when a branch or tag is created.                          |
| `delete`              | Triggered when a branch or tag is deleted.                          |
| `fork`                | When someone forks the repository.                                  |
| `star`                | When someone stars the repository.                                  |
| `watch`               | When someone watches the repository.                                |
| `workflow_run`        | Triggers when another workflow completes.                           |
| `repository_dispatch` | Custom webhook trigger.                                             |
| `check_run`           | Triggered by GitHub App for CI checks.                              |
| `deployment`          | Triggered when a deployment is created.                             |
| `deployment_status`   | Triggered when deployment status is updated.                        |

---

### **2. Sample Usage of Triggers in a Workflow**

```yaml
name: CI Pipeline

on:
  push:
    branches: [ "main" ]
    paths:
      - "**.java"
  pull_request:
    branches: [ "main" ]
  schedule:
    - cron: '0 3 * * *' # Runs daily at 3 AM UTC
  workflow_dispatch: # Manual trigger

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: ./gradlew test
```

---

### **3. Special Triggers**

* **workflow\_dispatch** (Manual Run)

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        default: 'staging'
```

* **repository\_dispatch** (Custom Webhook)

```yaml
on:
  repository_dispatch:
    types: [custom-event]
```

Trigger using a REST API call:

```bash
curl -X POST \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/OWNER/REPO/dispatches \
  -d '{"event_type":"custom-event"}'
```

---

Would you like a downloadable YAML cheat sheet with all events and example usage?
