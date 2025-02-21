### **Service Account in Kubernetes**  

A **Service Account** in Kubernetes is an identity used by **pods** to interact with the **Kubernetes API server**. It provides authentication for pods, allowing them to perform specific actions on cluster resources based on assigned **RBAC roles**.

---

## **🔹 Why Use Service Accounts?**
1. **Secure API Access:** Pods need to authenticate to perform actions like creating/deleting resources.
2. **Fine-Grained Permissions:** Using **RBAC**, you can restrict what a pod can do.
3. **Avoid Using Default Accounts:** By default, Kubernetes assigns a **default service account**, which has limited permissions.
4. **Separate Identities for Applications:** Different applications within the same cluster can have different service accounts with restricted access.

---

## **🔹 How Service Accounts Work**
- Every pod runs inside a **namespace** and gets a default service account (`default`).
- Kubernetes automatically mounts a token inside the pod at `/var/run/secrets/kubernetes.io/serviceaccount/` for authentication.
- The service account **does not** provide direct user access but is meant for applications running inside pods.

---

## **🔹 Creating a Custom Service Account**
By default, pods use the `default` service account. You can create a new one for better security.

### **Step 1: Define a Service Account**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app-sa
  namespace: my-namespace
```
🔹 **This service account (`my-app-sa`) is created in the `my-namespace` namespace.**

### **Step 2: Assign Permissions Using a Role**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: my-namespace
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
```
🔹 **This `Role` allows reading pods in `my-namespace`.**

### **Step 3: Bind the Role to the Service Account**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-reader-binding
  namespace: my-namespace
subjects:
  - kind: ServiceAccount
    name: my-app-sa
    namespace: my-namespace
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```
🔹 **Now, `my-app-sa` can only read pods in `my-namespace`.**

### **Step 4: Use the Service Account in a Pod**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  namespace: my-namespace
spec:
  serviceAccountName: my-app-sa
  containers:
    - name: my-container
      image: my-app:latest
```
🔹 **This pod will now authenticate using `my-app-sa`.**

---

## **🔹 Service Account vs. Namespace**
| Feature | Service Account | Namespace |
|---------|----------------|-----------|
| **Definition** | Identity for pods to interact with Kubernetes API | Logical separation of resources |
| **Scope** | Exists within a namespace | Contains multiple resources like pods, services, and roles |
| **Purpose** | Provides authentication & RBAC for pods | Isolates workloads and resources |
| **Example** | `my-app-sa` (used by a pod) | `dev`, `prod`, `staging` |

✅ **Namespace organizes resources**  
✅ **Service Account controls pod API access**

Would you like to see how to use service accounts for external authentication? 🚀