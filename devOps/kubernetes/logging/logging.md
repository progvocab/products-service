### **🔹 Integrating Logging & Monitoring for OIDC-Enabled Kubernetes (Kind + AWS Cognito)**
To ensure observability in your **OIDC-enabled Kind cluster**, we’ll integrate:  
✅ **Prometheus** for metrics collection  
✅ **Grafana** for visualization  
✅ **AWS CloudWatch** for centralized logs  

---

## **🔹 1. Deploy Prometheus & Grafana in Kubernetes**
### **1️⃣ Install Prometheus & Grafana Using Helm**
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack
```

### **2️⃣ Expose Prometheus & Grafana**
```bash
kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090
kubectl port-forward svc/prometheus-grafana 3000
```
- **Prometheus Dashboard**: `http://localhost:9090`  
- **Grafana Dashboard**: `http://localhost:3000`  
  - **Username:** `admin`
  - **Password:** `prom-operator`

### **3️⃣ Add Prometheus Data Source in Grafana**
- Go to **Settings → Data Sources**
- Select **Prometheus**
- Set **URL**: `http://prometheus-kube-prometheus-prometheus:9090`
- **Save & Test**

---

## **🔹 2. Send Kubernetes Logs to AWS CloudWatch**
### **1️⃣ Deploy Fluent Bit for Log Collection**
```bash
helm install fluent-bit stable/fluent-bit \
  --set cloudWatch.region=<AWS_REGION> \
  --set cloudWatch.logGroup=Kubernetes-Logs
```

### **2️⃣ Verify Logs in AWS CloudWatch**
```bash
aws logs describe-log-groups --query "logGroups[*].logGroupName"
aws logs tail Kubernetes-Logs --follow
```

---

## **🔹 3. Monitor OIDC Authentication & RBAC Logs**
### **1️⃣ Enable API Server Auditing**
Modify your **Kind cluster config**:
```yaml
apiServer:
  extraArgs:
    audit-log-path: "/var/log/kubernetes/audit.log"
    audit-log-maxage: "30"
    audit-log-maxsize: "100"
    audit-log-maxbackup: "5"
```

### **2️⃣ Query Authentication Logs in CloudWatch**
```bash
aws logs filter-log-events --log-group-name Kubernetes-Logs --filter-pattern '{ $.user.username = "kind-user" }'
```

---

## **🔹 Summary**
✅ **Prometheus & Grafana for monitoring**  
✅ **AWS CloudWatch for Kubernetes logs**  
✅ **Auditing OIDC authentication & RBAC events**  

Would you like **alerts for failed logins & security incidents?** 🚀

### **🔹 Setting Up Alerts for Failed Logins & Security Incidents in Kubernetes**  
We will configure **alerting mechanisms** for:  
✅ **Failed OIDC logins** (Authentication failures)  
✅ **Unauthorized Kubernetes API access** (RBAC violations)  
✅ **Unusual activity** (e.g., too many login attempts)

---

## **🔹 1. Configure Prometheus Alerting**
### **1️⃣ Create Alert Rules for Failed Logins**
- Failed logins can be detected using **OIDC authentication logs** from API Server.
- Create a Prometheus rule to **detect excessive 401/403 responses**.

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: failed-login-alerts
spec:
  groups:
  - name: failed-login-rules
    rules:
    - alert: FailedOIDCLogin
      expr: rate(apiserver_request_total{code="401"}[5m]) > 5
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "Multiple failed OIDC login attempts detected"
        description: "More than 5 failed logins in 5 minutes. Possible brute force attack."
```
**Apply the Rule**  
```bash
kubectl apply -f failed-login-alerts.yaml
```

---

## **🔹 2. Set Up Alertmanager to Send Notifications**
### **1️⃣ Install Alertmanager via Helm**
```bash
helm install alertmanager prometheus-community/prometheus \
  --set alertmanager.enabled=true
```

### **2️⃣ Configure AlertManager to Send Email/Slack Alerts**
Create `alertmanager-config.yaml`:
```yaml
global:
  resolve_timeout: 5m
route:
  receiver: "slack-alerts"
  repeat_interval: 30m
receivers:
  - name: "slack-alerts"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/XXXXXXXXX"
        channel: "#security-alerts"
        title: "OIDC Login Alert"
        text: "{{ .CommonAnnotations.summary }}"
```
**Apply Configuration**  
```bash
kubectl apply -f alertmanager-config.yaml
```

---

## **🔹 3. Monitor Unauthorized API Access via AWS CloudWatch**
### **1️⃣ Create a CloudWatch Alarm for Unauthorized Access**
Create a metric filter for **403 Forbidden responses** in the API server logs:
```bash
aws logs put-metric-filter --log-group-name Kubernetes-Logs \
  --filter-name UnauthorizedAccessFilter \
  --metric-transformations metricName=UnauthorizedAccess,metricNamespace=Kubernetes,metricValue=1 \
  --filter-pattern '{ $.code = "403" }'
```

### **2️⃣ Set Up an Alarm for Too Many Unauthorized Requests**
```bash
aws cloudwatch put-metric-alarm --alarm-name "TooManyUnauthorizedRequests" \
  --metric-name "UnauthorizedAccess" --namespace "Kubernetes" \
  --statistic Sum --period 300 --threshold 10 --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 --alarm-actions <SNS_TOPIC_ARN>
```

---

## **🔹 4. (Optional) Block Suspicious IPs Using Network Policies**
If brute force attacks are detected, you can block the source IP in **Kubernetes Network Policies**:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-suspicious-ips
spec:
  podSelector: {}
  ingress:
  - from:
    - ipBlock:
        cidr: "192.168.1.100/32"
```
**Apply Policy:**  
```bash
kubectl apply -f block-suspicious-ips.yaml
```

---

## **🔹 Summary**
✅ **Prometheus & AlertManager detect failed logins**  
✅ **Slack/email alerts for security incidents**  
✅ **CloudWatch alarm for unauthorized API access**  
✅ **Kubernetes Network Policies to block suspicious IPs**  

Would you like **IAM-based access control for Kubernetes with AWS IAM Roles?** 🚀