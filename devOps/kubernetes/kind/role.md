### **Understanding `Role` Kind in Kubernetes with Use Cases and Examples**  

In Kubernetes, a **Role** is an RBAC (Role-Based Access Control) object that defines permissions within a **specific namespace**. A **RoleBinding** assigns a Role to users, groups, or service accounts.

🔹 **Use Cases of Role:**
1. Restricting users to manage only Pods in a namespace.  
2. Allowing access to ConfigMaps but not Secrets.  
3. Granting only read permissions to specific users.  
4. Restricting a service account to access a particular resource.  

---

## **1️⃣ Create a Role for Managing Pods**
This Role allows a user to create, delete, list, and update **only Pods** in the `dev-environment` namespace.

### **`role-pod-manager.yaml`**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev-environment
  name: pod-manager
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "create", "delete"]
```
🔹 **Permissions:**
- ✅ Allowed: Managing Pods
- ❌ Denied: ConfigMaps, Secrets, Services

### **Bind Role to a User (`developer1`)**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-manager-binding
  namespace: dev-environment
subjects:
  - kind: User
    name: developer1
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-manager
  apiGroup: rbac.authorization.k8s.io
```
🔹 Now, `developer1` can manage Pods but cannot modify ConfigMaps, Secrets, or Deployments.

---

## **2️⃣ Create a Role for Read-Only Access**
A **read-only user** should only be able to view resources but not modify them.

### **`role-read-only.yaml`**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev-environment
  name: read-only
rules:
  - apiGroups: [""]
    resources: ["pods", "configmaps", "services"]
    verbs: ["get", "list"]
```
🔹 **Permissions:**
- ✅ Can **only view** Pods, ConfigMaps, and Services
- ❌ Cannot create, delete, or update any resource

### **Bind Role to a User (`viewer1`)**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-only-binding
  namespace: dev-environment
subjects:
  - kind: User
    name: viewer1
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: read-only
  apiGroup: rbac.authorization.k8s.io
```
🔹 Now, `viewer1` can only **view** resources but cannot make any changes.

---

## **3️⃣ Restrict ConfigMap Access but Allow Pod Management**
Some users should be able to manage Pods but should not have access to **ConfigMaps and Secrets**.

### **`role-pod-no-configmap.yaml`**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev-environment
  name: pod-manager-no-configmap
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "create", "delete"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: []  # No access to ConfigMaps
```
🔹 **Permissions:**
- ✅ Can manage Pods
- ❌ No access to ConfigMaps or Secrets

### **Bind Role to a User (`developer2`)**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-no-configmap-binding
  namespace: dev-environment
subjects:
  - kind: User
    name: developer2
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-manager-no-configmap
  apiGroup: rbac.authorization.k8s.io
```
🔹 Now, `developer2` can work with Pods but **cannot read or modify ConfigMaps**.

---

## **4️⃣ Grant Access to a Service Account (App)**
A service account should be allowed to read a specific ConfigMap but not modify it.

### **`role-app-configmap-reader.yaml`**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev-environment
  name: configmap-reader
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["app-config"]  # Only allow access to "app-config"
    verbs: ["get", "list"]
```
🔹 **Permissions:**
- ✅ Can only read **app-config ConfigMap**
- ❌ Cannot modify ConfigMaps or read other resources

### **Bind Role to a Service Account (`my-app-sa`)**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: configmap-reader-binding
  namespace: dev-environment
subjects:
  - kind: ServiceAccount
    name: my-app-sa  # Service Account of the application
    namespace: dev-environment
roleRef:
  kind: Role
  name: configmap-reader
  apiGroup: rbac.authorization.k8s.io
```
🔹 Now, `my-app-sa` can **read only `app-config` ConfigMap** but cannot read other resources.

---

## **5️⃣ Verify Access**
To check what a user can do:
```sh
kubectl auth can-i get pods --as=developer1 --namespace=dev-environment
kubectl auth can-i list configmaps --as=viewer1 --namespace=dev-environment
kubectl auth can-i delete secrets --as=developer2 --namespace=dev-environment
```
✅ If unauthorized, access is successfully restricted.

---

## **6️⃣ Summary Table**
| **User/Service** | **Allowed Resources** | **Restricted Resources** |
|-----------------|----------------------|------------------------|
| **developer1**  | Pods (CRUD) | ConfigMaps, Secrets |
| **viewer1**  | Read-Only (Pods, ConfigMaps, Services) | Cannot Modify Anything |
| **developer2**  | Pods (CRUD) | ConfigMaps, Secrets |
| **my-app-sa** (Service Account) | Read-Only Access to `app-config` | Cannot Access Other ConfigMaps |

✅ Now you understand **Roles in Kubernetes** and how they **restrict access to resources within a namespace**! 🚀  
Would you like to explore **ClusterRole for cluster-wide permissions**?


### **Restricting Access to ConfigMaps and Secrets in Kubernetes using RBAC**  

In Kubernetes, **ConfigMaps** and **Secrets** contain sensitive data like environment variables, API keys, and database credentials. To **restrict access to these resources**, we use **Role-Based Access Control (RBAC)**.

---

## **1️⃣ Define Access Requirements**  
We want to:
✅ **Allow developers** to manage pods but **deny access to ConfigMaps & Secrets**.  
✅ **Allow only admins** to access Secrets.  
✅ **Ensure service accounts for applications can access specific Secrets** but not all.

---

## **2️⃣ Create an RBAC Role to Restrict ConfigMap & Secret Access**  
The following **Role** allows users to **only view pods and services**, but denies them access to ConfigMaps and Secrets.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev-environment
  name: developer-role
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list", "create", "update", "delete"]
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: []  # No verbs = No access
```
Apply this role:
```sh
kubectl apply -f developer-role.yaml
```
✅ **Developers can now manage pods and services but cannot access ConfigMaps & Secrets.**

---

## **3️⃣ Assign Role to a Developer Using RoleBinding**
Now, we bind the `developer-role` to a specific user, `dev-user`.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-binding
  namespace: dev-environment
subjects:
  - kind: User
    name: dev-user  # Change this to the actual username
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: developer-role
  apiGroup: rbac.authorization.k8s.io
```
Apply this RoleBinding:
```sh
kubectl apply -f developer-binding.yaml
```
✅ **Now, `dev-user` cannot view or modify ConfigMaps or Secrets.**

---

## **4️⃣ Create an RBAC Role for Admins to Access Secrets**
Admins need access to **Secrets and ConfigMaps**, so we create a separate **admin role**.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev-environment
  name: admin-role
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "create", "update", "delete"]
```
Now bind it to an admin:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: admin-binding
  namespace: dev-environment
subjects:
  - kind: User
    name: admin-user
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: admin-role
  apiGroup: rbac.authorization.k8s.io
```
Apply it:
```sh
kubectl apply -f admin-binding.yaml
```
✅ **Now, `admin-user` has full access to Secrets and ConfigMaps.**

---

## **5️⃣ Restrict Access to Specific Secrets for a Service Account**
Sometimes, you need to allow **only certain apps** to access specific Secrets. This is done by binding a role to a **service account**.

### **Step 1: Create a Role That Grants Access to Only One Secret**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev-environment
  name: app-secret-access
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["database-secret"]  # Only allow access to "database-secret"
    verbs: ["get"]
```

---

### **Step 2: Bind the Role to the Application's Service Account**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-secret-binding
  namespace: dev-environment
subjects:
  - kind: ServiceAccount
    name: my-app-sa  # Service Account of the application
    namespace: dev-environment
roleRef:
  kind: Role
  name: app-secret-access
  apiGroup: rbac.authorization.k8s.io
```
Apply these:
```sh
kubectl apply -f app-secret-access.yaml
kubectl apply -f app-secret-binding.yaml
```
✅ **Now, only `my-app-sa` can access `database-secret`, while other applications cannot.**

---

## **6️⃣ Verify Access Permissions**
To check which permissions a user or service account has:
```sh
kubectl auth can-i get secrets --as=dev-user --namespace=dev-environment
kubectl auth can-i list configmaps --as=admin-user --namespace=dev-environment
```
To test application access:
```sh
kubectl exec -it my-app-pod --namespace=dev-environment -- cat /var/run/secrets/kubernetes.io/serviceaccount/token
```
If unauthorized, it means the restrictions are working.

---

## **7️⃣ Summary Table**
| **User/Service** | **Allowed Resources** | **Restricted Resources** |
|----------------|----------------|----------------|
| **Developer (`dev-user`)** | Pods, Services | ❌ ConfigMaps, Secrets |
| **Admin (`admin-user`)** | ConfigMaps, Secrets | ✅ Full access |
| **App (`my-app-sa`)** | Only `database-secret` | ❌ Other Secrets |

✅ **Now your Kubernetes cluster restricts ConfigMaps & Secrets access securely!** 🚀  
Would you like to integrate **audit logging or enforce additional security like encryption**?


