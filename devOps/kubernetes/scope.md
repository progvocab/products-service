In **Kubernetes**, every resource (kind) has a **scope**, which determines whether it operates at the **Cluster** level or the **Namespace** level.  

---

# **🔹 Scope of Kubernetes Resources**
There are **two main scopes**:  
1. **Namespace-scoped** → Applies within a specific **namespace**.  
2. **Cluster-scoped** → Applies to the **entire cluster**.

---

# **🔹 Namespace-Scoped Resources**
These resources exist **inside a namespace** and cannot be referenced outside of it.

| **Resource (Kind)**  | **Scope** | **Description** |
|----------------------|-----------|----------------|
| **Pod** | Namespace | A running container instance. |
| **Deployment** | Namespace | Manages a set of replica pods. |
| **StatefulSet** | Namespace | Manages stateful applications. |
| **DaemonSet** | Namespace | Ensures a pod runs on every node. |
| **ReplicaSet** | Namespace | Ensures a specified number of pods run. |
| **Job** | Namespace | Runs a one-time batch task. |
| **CronJob** | Namespace | Schedules periodic jobs. |
| **Service** | Namespace | Exposes applications within a namespace. |
| **ConfigMap** | Namespace | Stores configuration data. |
| **Secret** | Namespace | Stores sensitive information. |
| **Ingress** | Namespace | Manages external access to services. |
| **PersistentVolumeClaim (PVC)** | Namespace | Requests storage from a PersistentVolume. |
| **Role** | Namespace | Defines permissions within a namespace. |
| **RoleBinding** | Namespace | Grants a Role's permissions within a namespace. |

**Example:**  
A **Pod** running in `namespace1` **cannot** access ConfigMaps from `namespace2`.

---

# **🔹 Cluster-Scoped Resources**
These resources exist at the **cluster level** and apply across all namespaces.

| **Resource (Kind)**  | **Scope** | **Description** |
|----------------------|-----------|----------------|
| **Node** | Cluster | Represents a worker node in the cluster. |
| **Namespace** | Cluster | Creates logical isolation within the cluster. |
| **PersistentVolume (PV)** | Cluster | A storage resource available to all namespaces. |
| **ClusterRole** | Cluster | Defines permissions at the cluster level. |
| **ClusterRoleBinding** | Cluster | Grants ClusterRole permissions across the cluster. |
| **StorageClass** | Cluster | Defines how storage is provisioned dynamically. |
| **CustomResourceDefinition (CRD)** | Cluster | Defines new custom resource types. |
| **APIService** | Cluster | Extends the Kubernetes API. |
| **MutatingWebhookConfiguration** | Cluster | Defines admission webhooks for modifying objects. |
| **ValidatingWebhookConfiguration** | Cluster | Defines webhooks for validation. |

**Example:**  
A **PersistentVolume** can be shared among multiple namespaces, but a **PersistentVolumeClaim** is namespace-scoped.

---

# **🔹 How to Check Scope of a Resource?**
You can check whether a Kubernetes resource is **namespace-scoped** or **cluster-scoped** using:
```sh
kubectl api-resources --namespaced=true   # Shows namespace-scoped resources
kubectl api-resources --namespaced=false  # Shows cluster-scoped resources
```

Would you like **examples of RBAC configurations for namespace and cluster scopes?** 🚀