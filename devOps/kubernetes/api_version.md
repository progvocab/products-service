### **🔹 API Versions in Kubernetes**
In Kubernetes, API resources are organized into **different API groups and versions**, which define their stability and availability.

---

# **🔹 API Versioning Format**
Kubernetes API versions follow this format:
```
<api-group>/<version>
```
- **`api-group`** → Organizes related resources.  
- **`version`** → Represents stability:  
  - `v1` → Stable, widely used.  
  - `v1beta1`, `v2alpha1` → Experimental or deprecated.

---

# **🔹 Common API Groups & Versions**
| **API Version**       | **Scope**       | **Example Resource** | **Description** |
|-----------------------|----------------|----------------------|----------------|
| `core/v1` (or `v1`)  | Core API (default) | `Pod`, `Service` | Basic Kubernetes objects. |
| `apps/v1`            | Workloads       | `Deployment`, `StatefulSet`, `DaemonSet`, `ReplicaSet` | Manages applications. |
| `batch/v1`           | Batch Jobs      | `Job`, `CronJob` | Handles batch workloads. |
| `autoscaling/v2`     | Autoscaling     | `HorizontalPodAutoscaler` | Automatically scales pods. |
| `rbac.authorization.k8s.io/v1` | Security & Access | `Role`, `ClusterRole`, `RoleBinding` | Role-based access control (RBAC). |
| `networking.k8s.io/v1` | Networking | `Ingress`, `NetworkPolicy` | Manages network policies & routing. |
| `storage.k8s.io/v1`  | Storage         | `StorageClass`, `VolumeSnapshot` | Handles persistent storage. |
| `policy/v1`          | Security Policy | `PodDisruptionBudget` | Defines pod availability policies. |

---

## **🔹 1. `core/v1` (or `v1`) → Default API Group**
The **core API group** (`v1`) has no prefix.

### **Example: Create a Pod (`v1`)**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: my-container
      image: nginx
```

### **Example: Create a Service (`v1`)**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: my-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```

---

## **🔹 2. `apps/v1` → Workload API Group**
Used for **Deployments, StatefulSets, DaemonSets**, and **ReplicaSets**.

### **Example: Create a Deployment (`apps/v1`)**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: nginx
        image: nginx:latest
```

---

## **🔹 3. `batch/v1` → Jobs & CronJobs**
Handles **batch workloads** (e.g., one-time jobs, scheduled tasks).

### **Example: Create a Job (`batch/v1`)**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: my-job
spec:
  template:
    spec:
      containers:
        - name: my-job-container
          image: busybox
          command: ["echo", "Hello, Kubernetes!"]
      restartPolicy: Never
```

### **Example: Create a CronJob (`batch/v1`)**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: my-cronjob
spec:
  schedule: "*/5 * * * *"  # Runs every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cron-container
            image: busybox
            command: ["echo", "Running scheduled task"]
          restartPolicy: OnFailure
```

---

## **🔹 4. `autoscaling/v2` → Horizontal Pod Autoscaling**
Used for **autoscaling pods** based on CPU/Memory.

### **Example: Create a Horizontal Pod Autoscaler (`autoscaling/v2`)**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 50
```

---

## **🔹 5. `rbac.authorization.k8s.io/v1` → RBAC API**
Manages **Role-Based Access Control (RBAC)**.

### **Example: Create a Role & RoleBinding (`rbac.authorization.k8s.io/v1`)**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: default
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-reader-binding
  namespace: default
subjects:
- kind: User
  name: alice
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

---

## **🔹 6. `networking.k8s.io/v1` → Ingress & Network Policies**
Manages **network access control**.

### **Example: Create an Ingress (`networking.k8s.io/v1`)**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-service
            port:
              number: 80
```

---

## **🔹 7. `storage.k8s.io/v1` → Persistent Storage**
Handles **PersistentVolumes (PVs) and StorageClasses**.

### **Example: Create a StorageClass (`storage.k8s.io/v1`)**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-storage
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp2
```

---

# **🔹 How to Check API Versions in Your Cluster?**
```sh
kubectl api-resources  # Lists all API resources & versions
kubectl explain deployment  # Shows the API version & fields for a resource
kubectl get deployments --api-version=apps/v1  # Uses a specific API version
```

---

# **🔹 How to Check Deprecated API Versions?**
Kubernetes **deprecates older APIs** over time.  
Run:
```sh
kubectl get --raw /openapi/v2 | jq '.definitions | keys' | grep v1beta
```
Or check Kubernetes release notes for deprecations:  
🔗 [https://kubernetes.io/docs/reference/using-api/deprecation-guide/](https://kubernetes.io/docs/reference/using-api/deprecation-guide/)

---

# **🔹 Final Thoughts**
- ✅ Use **`v1`** for core resources (Pods, Services).
- ✅ Use **`apps/v1`** for workloads (Deployments, StatefulSets).
- ✅ Use **`batch/v1`** for **Jobs & CronJobs**.
- ✅ Use **`rbac.authorization.k8s.io/v1`** for security.
- ✅ Use **`networking.k8s.io/v1`** for network policies.
- ✅ Always check for **API deprecations** in Kubernetes updates.

Would you like examples for **custom resources (CRDs) or Kubernetes API calls**? 🚀