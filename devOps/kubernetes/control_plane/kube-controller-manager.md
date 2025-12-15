In Kubernetes, **“controller manager”** refers to control-plane components that run **control loops** to move the cluster toward the desired state stored in **etcd**.
There are **two distinct controller managers**.

 

### **1. kube-controller-manager**

**Count:** 1 logical component (runs many controllers inside it)

**What it does (3 lines):**

* Runs core Kubernetes controllers like Node, ReplicaSet, Deployment, Job, Endpoint, and Namespace controllers.
* Watches API Server for desired state and reconciles actual cluster state continuously.
* Handles lifecycle operations such as pod replication, node health, and garbage collection.

**Where it lives:**

* Runs as a **static Pod** or systemd service on **control-plane nodes**
* Namespace: `kube-system`
* Communicates with **API Server** and **etcd**

 

### **2. cloud-controller-manager**

**Count:** 1 logical component (cloud-specific, optional)

**What it does (3 lines):**

* Integrates Kubernetes with the underlying cloud provider (AWS, GCP, Azure, etc.).
* Manages cloud resources like LoadBalancers, Nodes, Routes, and Volumes.
* Separates cloud logic from core Kubernetes to keep Kubernetes cloud-agnostic.

**Where it lives:**

* Runs as a **Pod on control-plane nodes**
* Namespace: `kube-system`
* Communicates with **API Server** and cloud APIs (not directly with kubelet)

 

### **Controllers Inside `kube-controller-manager`**

*(These are NOT separate processes, but internal controllers)*

| Controller                        | Responsibility                          |
| --------------------------------- | --------------------------------------- |
| Node Controller                   | Monitors node health and eviction       |
| ReplicaSet Controller             | Maintains desired pod replicas          |
| Deployment Controller             | Manages rolling updates & rollbacks     |
| Job Controller                    | Manages batch and parallel jobs         |
| CronJob Controller                | Schedules Jobs                          |
| Endpoint/EndpointSlice Controller | Maintains Service endpoints             |
| Namespace Controller              | Cleans up resources on namespace delete |
| ServiceAccount Controller         | Creates default service accounts        |
| Garbage Collector                 | Deletes orphaned resources              |
| PV Controller                     | Manages PersistentVolume lifecycle      |



 ```mermaid
flowchart TD
    A[kubectl] -->|REST request| B[API Server]

    B -->|Validate & AuthZ| B
    B -->|Persist desired state| C[etcd]

    C -->|Watch / Change Stream| D[Kube Controller Manager]

    D -->|Reconciliation Loop| E[Deployment Controller]

    E -->|Create / Update| F[ReplicaSet]

    F -->|Create Pods| G[Pod Objects]

    G -->|Scheduling Needed| H[Kube Scheduler]

    H -->|Bind Pod to Node| B

    B -->|Updated State| C

    C -->|Watch| I[Kubelet]

    I -->|Create Containers| J[Container Runtime]

    J -->|Running Pods| K[Application Running]
```

### **Control Plane Placement Summary**

| Component                | Location                       |
| ------------------------ | ------------------------------ |
| kube-apiserver           | Control-plane nodes            |
| etcd                     | Control-plane nodes            |
| kube-scheduler           | Control-plane nodes            |
| kube-controller-manager  | Control-plane nodes            |
| cloud-controller-manager | Control-plane nodes (optional) |

 

### **Key Exam / Interview Takeaway**

* **Controller Managers = 2**
* **kube-controller-manager** → core Kubernetes logic
* **cloud-controller-manager** → cloud integration
* Controllers run **control loops**, not one-time actions
* They **never talk directly to Pods**, only via the API Server

More : 

* Mermaid diagram of control-plane interaction
* List of controllers mapped to CRDs
* Difference between controllers and operators
* How leader election works for controller managers in HA setups
