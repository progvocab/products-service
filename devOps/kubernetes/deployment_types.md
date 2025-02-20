# **Kubernetes Deployment Strategies**  

When deploying applications in **Kubernetes**, choosing the right strategy ensures **zero downtime, rollback capabilities, and efficient resource utilization**. Below are the most commonly used deployment strategies.

---

## **1️⃣ Rolling Update (Default)**
🔹 **Gradually replaces old pods with new ones**  
🔹 **No downtime**  
🔹 Can be controlled using **maxSurge** and **maxUnavailable**  

### **📌 How It Works**
- New pods are gradually created while old ones are terminated.
- Ensures **zero downtime** by maintaining a minimum number of running instances.
- Suitable for **most applications** that can handle incremental updates.

### **📌 Example Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rolling-update-demo
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Extra pod allowed during update
      maxUnavailable: 1  # Max pods that can be unavailable
  selector:
    matchLabels:
      app: demo
  template:
    metadata:
      labels:
        app: demo
    spec:
      containers:
      - name: demo-container
        image: my-app:v2  # New version of the app
```
✅ **Best For:** Most applications, ensuring smooth updates without downtime.  
❌ **Drawback:** If something goes wrong, rolling back can take time.

---

## **2️⃣ Blue-Green Deployment**
🔹 **Two identical environments (Blue & Green)**  
🔹 **Instant rollback if issues occur**  
🔹 **Uses a Service to switch traffic between versions**  

### **📌 How It Works**
1. **Blue** represents the live version.
2. **Green** is deployed with the new version.
3. Traffic is switched from **Blue** to **Green** using a **Kubernetes Service**.
4. If something goes wrong, traffic can be reverted to **Blue**.

### **📌 Example Deployment**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app-green  # Switch to new version dynamically
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```
✅ **Best For:** Critical applications requiring zero downtime and easy rollback.  
❌ **Drawback:** Requires **double resources**, which can be costly.

---

## **3️⃣ Canary Deployment**
🔹 **Deploys new version to a small subset of users**  
🔹 **Progressively increases rollout if stable**  
🔹 Uses **labels and selectors** to manage traffic  

### **📌 How It Works**
1. Deploy **new version** to a small percentage of users.
2. Monitor for failures (e.g., using Prometheus).
3. Gradually shift more traffic to the new version.
4. If issues arise, roll back easily.

### **📌 Example Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: canary-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app-container
        image: my-app:v2  # Canary version
```
✅ **Best For:** Applications requiring gradual exposure to minimize risks.  
❌ **Drawback:** Needs **traffic management (e.g., Istio, Nginx Ingress)** for routing.

---

## **4️⃣ Recreate Deployment**
🔹 **Stops old version before deploying the new one**  
🔹 **Downtime occurs during deployment**  
🔹 **Simple and resource-efficient**  

### **📌 How It Works**
1. All existing **pods are terminated**.
2. New pods with the updated version are created.
3. There is a **brief downtime** between stopping old pods and starting new ones.

### **📌 Example Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recreate-deployment
spec:
  strategy:
    type: Recreate  # Stops old pods before creating new ones
  selector:
    matchLabels:
      app: demo
  template:
    metadata:
      labels:
        app: demo
    spec:
      containers:
      - name: demo-container
        image: my-app:v2
```
✅ **Best For:** Stateless applications where downtime is acceptable.  
❌ **Drawback:** Causes **downtime**, making it unsuitable for high-availability services.

---

## **5️⃣ A/B Testing Deployment**
🔹 **Routes traffic based on user attributes (e.g., location, device, user group)**  
🔹 **Requires advanced traffic management (e.g., Istio, Nginx Ingress)**  
🔹 **Can test different features for different users**  

### **📌 How It Works**
1. Users are split into different groups (e.g., **Group A gets v1, Group B gets v2**).
2. Routing happens **at the load balancer level**.
3. Telemetry data is analyzed to decide the success of the new version.

✅ **Best For:** AI-based features, feature flag testing, and gradual rollouts.  
❌ **Drawback:** Needs **advanced traffic control tools**.

---

## **Comparison Table**

| Strategy          | **Downtime** | **Rollback** | **Resource Usage** | **Use Case** |
|------------------|------------|-------------|----------------|-------------|
| Rolling Update   | No         | Slow rollback | Efficient | Most applications |
| Blue-Green      | No         | Instant | High (duplicate resources) | Critical applications |
| Canary          | No         | Fast | Moderate | Gradual rollouts with monitoring |
| Recreate        | Yes        | Fast | Low | Simple apps where downtime is okay |
| A/B Testing     | No         | Fast | High (requires traffic splitting) | Feature flag testing |

---

### **Which Deployment Strategy Should You Use?**
- **For zero-downtime, safe updates → Rolling Update**
- **For instant rollback & critical apps → Blue-Green**
- **For gradual testing & exposure → Canary**
- **For simple, non-critical apps → Recreate**
- **For user-based experiments → A/B Testing**

Would you like real-world **case studies** or **hands-on Kubernetes commands** for these strategies? 🚀