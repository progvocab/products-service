### **What is Kubelet in Kubernetes?**  

Kubelet is a **node agent** in Kubernetes that runs on every **worker node**. It ensures that containers in **Pods** are running and remain healthy.  

---
![Kubelet](/assets/Kubelet.png)


## **1. Key Functions of Kubelet**
| **Function** | **Description** |
|-------------|----------------|
| **Pod Management** | Ensures that containers inside Pods run as expected. |
| **Container Runtime Communication** | Interacts with **Docker, containerd, or CRI-O** to start and stop containers. |
| **Health Monitoring** | Checks Pod & container health and restarts them if they fail. |
| **Syncs with Control Plane** | Fetches Pod definitions from the **API Server** and ensures the desired state is maintained. |
| **Logs & Metrics Collection** | Sends node & container logs to monitoring tools. |

---

## **2. How Kubelet Works (Example)**
### **Scenario: Running an Nginx Pod**  

1️⃣ **User Creates a Pod**  
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  containers:
  - name: nginx-container
    image: nginx:latest
```
- This Pod specification is **sent to the Kubernetes API Server**.

2️⃣ **Kubelet Receives the Pod Spec**  
- The **API Server** assigns the Pod to a worker node.  
- The **Kubelet on that node** pulls the container image and starts it using the **container runtime**.  

3️⃣ **Kubelet Monitors Pod Health**  
- If the container crashes, Kubelet restarts it.  
- If a liveness probe fails, Kubelet kills and restarts the container.  

---

## **3. Checking Kubelet Logs & Status**
To check if Kubelet is running on a node:  
```bash
systemctl status kubelet
```

To check Kubelet logs for debugging:  
```bash
journalctl -u kubelet -f
```

---

## **4. Kubelet vs. Other Kubernetes Components**
| **Component** | **Role** |
|--------------|---------|
| **Kubelet** | Runs on worker nodes, manages Pods & containers. |
| **Kube-Proxy** | Manages networking between Pods & Services. |
| **Kube-Scheduler** | Assigns Pods to worker nodes. |
| **Kube-Controller-Manager** | Handles node lifecycle, replication, etc. |

---

## **5. Summary**
✔ Kubelet runs on **each worker node** and ensures Pods run correctly.  
✔ It interacts with the **API Server** and **container runtime** (e.g., Docker).  
✔ It restarts failed Pods and enforces **health checks**.  

Would you like to learn about **Kubelet health probes** (Liveness, Readiness, Startup)?
