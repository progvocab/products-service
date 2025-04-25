To automate the execution of a Python script daily using Kubernetes, you can use a **Kubernetes CronJob**. A **CronJob** in Kubernetes allows you to schedule and run jobs periodically like a traditional **cron job** in Linux.

---

## **Steps to Automate a Python Script Daily Using Kubernetes**
1. **Create a Docker Image** of your Python script.
2. **Push the image** to a container registry (Docker Hub, AWS ECR, GCR, etc.).
3. **Define a Kubernetes CronJob** YAML file.
4. **Apply the CronJob** to your Kubernetes cluster.

---

### **Step 1: Create a Python Script**
Save this Python script as `script.py`:
```python
# script.py
from datetime import datetime

def main():
    print(f"Python script executed at: {datetime.now()}")

if __name__ == "__main__":
    main()
```

---

### **Step 2: Create a Dockerfile**
Create a `Dockerfile` to containerize your script:
```dockerfile
# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy the Python script into the container
COPY script.py .

# Run the script
CMD ["python", "script.py"]
```

---

### **Step 3: Build and Push the Docker Image**
Run the following commands:
```sh
# Build the Docker image
docker build -t your-dockerhub-username/python-script .

# Push the image to Docker Hub
docker push your-dockerhub-username/python-script
```
Replace `your-dockerhub-username` with your actual **Docker Hub username**.

---

### **Step 4: Define a Kubernetes CronJob**
Create a file called `cronjob.yaml`:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-python-script
spec:
  schedule: "0 0 * * *"  # Runs daily at midnight UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: python-script
            image: your-dockerhub-username/python-script:latest
            imagePullPolicy: Always
          restartPolicy: OnFailure
```
**Explanation:**
- `schedule: "0 0 * * *"` → Runs daily at **midnight UTC**.
- `restartPolicy: OnFailure` → Retries if the script fails.
- `imagePullPolicy: Always` → Ensures the latest image is used.

---

### **Step 5: Apply the CronJob to Kubernetes**
Run the following command:
```sh
kubectl apply -f cronjob.yaml
```
Verify if the CronJob is scheduled:
```sh
kubectl get cronjob
```
Check completed jobs:
```sh
kubectl get jobs
```
View logs of the latest job:
```sh
kubectl logs --selector=job-name=<job-name>
```

---

## **Bonus: Customizing the Schedule**
Modify the `schedule` field in `cronjob.yaml`:
| Frequency | Cron Expression |
|-----------|----------------|
| Every 5 mins | `*/5 * * * *` |
| Every hour | `0 * * * *` |
| Every day at 3 AM | `0 3 * * *` |
| Every Sunday at midnight | `0 0 * * 0` |

Would you like **email notifications** for job failures or retries? 🚀

To receive **email notifications** for failures or retries in a **Kubernetes CronJob**, you can integrate Kubernetes **Event Monitoring** with an **SMTP-based notification system** using **Prometheus Alertmanager** or **a simple Python script** in Kubernetes.

---

# **Approach 1: Using Kubernetes Events & Python SMTP**
We can create a **Kubernetes watcher** script that listens for failed jobs and sends an email.

## **1️⃣ Create a Python Script for Email Alerts**
Save this as `email_alert.py`:
```python
import smtplib
from email.mime.text import MIMEText
import subprocess
import json

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"  # Change if using another provider
SMTP_PORT = 587
EMAIL_SENDER = "your-email@gmail.com"
EMAIL_PASSWORD = "your-app-password"  # Use an App Password instead of your real password
EMAIL_RECEIVER = "receiver-email@gmail.com"

def get_failed_jobs():
    """Fetch failed Kubernetes jobs"""
    cmd = "kubectl get jobs -o json"
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    jobs = json.loads(output.stdout)

    failed_jobs = []
    for job in jobs.get("items", []):
        name = job["metadata"]["name"]
        status = job.get("status", {})
        failed = status.get("failed", 0)
        if failed > 0:
            failed_jobs.append(name)
    
    return failed_jobs

def send_email(failed_jobs):
    """Send an email alert"""
    subject = "Kubernetes Job Failure Alert 🚨"
    body = f"The following Kubernetes jobs have failed: \n\n" + "\n".join(failed_jobs)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

if __name__ == "__main__":
    failed_jobs = get_failed_jobs()
    if failed_jobs:
        send_email(failed_jobs)
```

### **How It Works:**
- It **checks Kubernetes for failed jobs** (`kubectl get jobs -o json`).
- If **failed jobs** exist, it **sends an email** via SMTP.
- The script runs **periodically** using a CronJob.

---

## **2️⃣ Deploy the Script as a Kubernetes CronJob**
Save this as `email-cronjob.yaml`:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: email-notifier
spec:
  schedule: "*/30 * * * *"  # Runs every 30 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: email-alert
            image: your-dockerhub-username/email-notifier:latest
            imagePullPolicy: Always
            env:
              - name: SMTP_SERVER
                value: "smtp.gmail.com"
              - name: SMTP_PORT
                value: "587"
              - name: EMAIL_SENDER
                value: "your-email@gmail.com"
              - name: EMAIL_PASSWORD
                value: "your-app-password"
              - name: EMAIL_RECEIVER
                value: "receiver-email@gmail.com"
          restartPolicy: OnFailure
```

### **Steps to Deploy:**
1. **Create a Docker Image**:
   ```sh
   docker build -t your-dockerhub-username/email-notifier .
   docker push your-dockerhub-username/email-notifier
   ```
2. **Apply the CronJob**:
   ```sh
   kubectl apply -f email-cronjob.yaml
   ```
3. **Check Logs**:
   ```sh
   kubectl logs --selector=job-name=email-notifier
   ```

---

# **Approach 2: Using Prometheus + Alertmanager**
For a **scalable** and **production-grade** solution, use **Prometheus Alertmanager** to monitor failed jobs.

## **Steps:**
1. **Install Prometheus & Alertmanager**:
   ```sh
   kubectl apply -f https://github.com/prometheus-operator/kube-prometheus/releases/download/v0.12.0/kube-prometheus-stack.yaml
   ```
2. **Configure Alert Rules (`prometheus-rules.yaml`)**:
   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: PrometheusRule
   metadata:
     name: job-failure-alerts
   spec:
     groups:
     - name: job-failure-alerts
       rules:
       - alert: KubernetesJobFailed
         expr: kube_job_failed > 0
         for: 5m
         labels:
           severity: critical
         annotations:
           summary: "Kubernetes Job Failed"
           description: "A Kubernetes job has failed."
   ```
3. **Configure Alertmanager (`alertmanager-config.yaml`)**:
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: alertmanager-config
   data:
     alertmanager.yml: |
       route:
         receiver: email
       receivers:
       - name: email
         email_configs:
         - to: "receiver-email@gmail.com"
           from: "your-email@gmail.com"
           smarthost: "smtp.gmail.com:587"
           auth_username: "your-email@gmail.com"
           auth_password: "your-app-password"
   ```
4. **Apply the Configurations**:
   ```sh
   kubectl apply -f prometheus-rules.yaml
   kubectl apply -f alertmanager-config.yaml
   ```

---

# **Which Approach Should You Choose?**
| Approach | Pros | Cons |
|----------|------|------|
| **Python SMTP Email Notifier** | ✅ Easy to set up <br> ✅ Works with any Kubernetes cluster <br> ✅ No external dependencies | ❌ Requires a separate CronJob <br> ❌ Less scalable |
| **Prometheus Alertmanager** | ✅ Scalable <br> ✅ Can send **Slack, PagerDuty, and Email alerts** <br> ✅ Ideal for production | ❌ Requires Prometheus setup <br> ❌ More complex |

---

# **Final Thoughts**
- For **small projects**, use **Python + SMTP** for quick email alerts.
- For **large-scale production** workloads, use **Prometheus + Alertmanager**.

Would you like to extend this with **Slack alerts or a webhook notification system**? 🚀

To extend **Kubernetes job failure alerts** to **Slack or Webhook notifications**, we can use **Prometheus Alertmanager** or a **Python script with webhooks**.

---

# **Approach 1: Slack Notifications via Prometheus Alertmanager**
### **1️⃣ Install Prometheus and Alertmanager**
If you haven't installed Prometheus, install it using:
```sh
kubectl apply -f https://github.com/prometheus-operator/kube-prometheus/releases/download/v0.12.0/kube-prometheus-stack.yaml
```

### **2️⃣ Create a Slack Incoming Webhook**
1. Go to [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks).
2. Create a new webhook and copy the URL.

### **3️⃣ Configure Alertmanager for Slack**
Save this as `alertmanager-config.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    route:
      receiver: "slack"
    receivers:
    - name: "slack"
      slack_configs:
      - channel: "#alerts"
        send_resolved: true
        api_url: "https://hooks.slack.com/services/your/slack/webhook"
        title: "🚨 Kubernetes Job Failure Alert!"
        text: "A job has failed in the cluster. Check logs immediately."
```
> **Replace** `"https://hooks.slack.com/services/your/slack/webhook"` with your actual Slack Webhook URL.

### **4️⃣ Apply the Configuration**
```sh
kubectl apply -f alertmanager-config.yaml
```

---

# **Approach 2: Slack Alerts via Python Webhook**
If you **don’t want to use Prometheus**, you can use a **Python script** that sends alerts via Slack.

### **1️⃣ Create `slack_alert.py`**
```python
import requests
import subprocess
import json

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/your/slack/webhook"

def get_failed_jobs():
    """Fetch failed Kubernetes jobs"""
    cmd = "kubectl get jobs -o json"
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    jobs = json.loads(output.stdout)

    failed_jobs = []
    for job in jobs.get("items", []):
        name = job["metadata"]["name"]
        status = job.get("status", {})
        failed = status.get("failed", 0)
        if failed > 0:
            failed_jobs.append(name)
    
    return failed_jobs

def send_slack_notification(failed_jobs):
    """Send Slack alert"""
    payload = {
        "text": f"🚨 *Kubernetes Job Failure Alert!* 🚨\n\nFailed Jobs:\n" + "\n".join(failed_jobs)
    }
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    if response.status_code == 200:
        print("✅ Slack notification sent!")
    else:
        print(f"❌ Failed to send Slack notification: {response.text}")

if __name__ == "__main__":
    failed_jobs = get_failed_jobs()
    if failed_jobs:
        send_slack_notification(failed_jobs)
```

### **2️⃣ Deploy the Script as a Kubernetes CronJob**
Save this as `slack-alert-cronjob.yaml`:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: slack-notifier
spec:
  schedule: "*/15 * * * *"  # Runs every 15 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: slack-alert
            image: your-dockerhub-username/slack-notifier:latest
            imagePullPolicy: Always
            env:
              - name: SLACK_WEBHOOK_URL
                value: "https://hooks.slack.com/services/your/slack/webhook"
          restartPolicy: OnFailure
```

### **3️⃣ Build and Push the Docker Image**
```sh
docker build -t your-dockerhub-username/slack-notifier .
docker push your-dockerhub-username/slack-notifier
```

### **4️⃣ Apply the CronJob**
```sh
kubectl apply -f slack-alert-cronjob.yaml
```

---

# **Approach 3: Webhook Notifications for Any Service**
If you need **custom webhooks** (e.g., Discord, PagerDuty, or a custom API), modify `slack_alert.py` like this:

```python
WEBHOOK_URL = "https://your-webhook-url.com"

def send_webhook_alert(failed_jobs):
    payload = {
        "alert": "Kubernetes Job Failure",
        "failed_jobs": failed_jobs
    }
    response = requests.post(WEBHOOK_URL, json=payload)
    print(response.status_code, response.text)
```

Use it in **any third-party service** by changing the `WEBHOOK_URL`.

---

# **Which Approach Should You Choose?**
| Approach | Pros | Cons |
|----------|------|------|
| **Slack via Prometheus Alertmanager** | ✅ Scalable <br> ✅ No coding required <br> ✅ Can integrate with other tools | ❌ Needs Prometheus setup |
| **Python Webhook Script** | ✅ Simple to implement <br> ✅ Works in any Kubernetes cluster | ❌ Requires a separate CronJob |
| **Custom Webhook Notifications** | ✅ Flexible (any service) <br> ✅ Works with Discord, PagerDuty, etc. | ❌ Requires API endpoint |

---

# **Final Thoughts**
- If **using Prometheus**, go with **Alertmanager**.
- If you **just need Slack alerts**, use the **Python CronJob**.
- If you need **a custom notification system**, use **webhooks**.

Would you like **integration with Discord, PagerDuty, or a monitoring dashboard**? 🚀