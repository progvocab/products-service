Kubernetes namespaces are logical partitions within a cluster that allow separation of resources, isolation, and organization. Let’s explore the specific namespaces you mentioned (`kube-flannel`, `node-lease`, `kube-public`, and `kube-system`) and their purposes:
 

### **1. `kube-flannel` Namespace**
- **Purpose**: 
  - This namespace is specific to the Flannel network plugin, which is a **CNI (Container Network Interface)** used for Kubernetes networking.
  - It manages Pods and components responsible for providing an overlay network that enables communication between Pods across nodes.
  
- **Key Components**:
  - **`flannel` DaemonSet**:
    - Runs on each node to configure the network and ensure proper Pod-to-Pod communication.
    - Assigns IP addresses to Pods using a virtual network.
  - **ConfigMap**:
    - Stores the Flannel configuration, such as CIDR ranges and backend types (e.g., VXLAN or host-gw).

- **Importance**:
  - If this namespace or the `flannel` Pods fail, inter-Pod communication across nodes will break.

 

### **2. `node-lease` Namespace**
- **Purpose**:
  - Used to store **Node Lease objects**, which are heartbeats sent by nodes to inform the control plane that they are healthy and still functioning.
  - These leases improve the performance of node health checks.

- **Key Components**:
  - **Lease Objects**:
    - Each node has a corresponding lease object in this namespace.
    - These objects are updated frequently (default is every 10 seconds) to indicate node liveness.

- **Importance**:
  - Enhances scalability and performance of large clusters by reducing the load on the `kube-apiserver`.
  - Prevents unnecessary eviction of Pods by ensuring timely node health checks.

 

### **3. `kube-public` Namespace**
- **Purpose**:
  - This namespace is automatically created by Kubernetes and is **readable by all users**, even unauthenticated ones.
  - Typically used for sharing publicly accessible cluster information.

- **Key Components**:
  - **Cluster Information ConfigMap**:
    - Contains cluster-level public information that may be useful for users (e.g., the cluster's API server address).

- **Common Use Cases**:
  - Sharing resources like public ConfigMaps.
  - Exposing cluster details that do not require authentication.
  
- **Example**:
  - The `kube-public` namespace may contain a ConfigMap with metadata about the cluster, which can be accessed by any user using:
    ```bash
    kubectl get configmap -n kube-public
    ```

- **Importance**:
  - Not critical for cluster functionality but useful for public information sharing.

 

### **4. `kube-system` Namespace**
- **Purpose**:
  - Houses the **system-level components** that are required to run and manage the Kubernetes cluster.
  - This namespace is automatically created during cluster initialization.

- **Key Components**:
  - **Control Plane Components** (if running as Pods):
    - `kube-apiserver`, `kube-controller-manager`, `kube-scheduler`.
  - **Networking Components**:
    - `kube-proxy` (manages network rules for services).
    - CoreDNS (handles DNS-based service discovery).
  - **Other System Add-Ons**:
    - Metrics server for resource monitoring.
    - Networking plugins like Flannel or Calico (if installed).
    - Cloud-specific controllers (e.g., for AWS or GCP).

- **Importance**:
  - The `kube-system` namespace is **critical** for cluster operation.
  - If Pods in this namespace fail, the cluster may stop functioning properly.

 
 

| **Namespace**    | **Purpose**                                                                                  | **Critical for Cluster?** |
|-------------------|----------------------------------------------------------------------------------------------|---------------------------|
| **`kube-flannel`**| Manages Flannel CNI components for inter-Pod networking.                                     | Yes, if using Flannel.    |
| **`node-lease`**  | Optimizes node heartbeat updates and improves scalability of node health checks.             | Yes.                      |
| **`kube-public`** | Stores publicly accessible information (e.g., cluster metadata).                             | No.                       |
| **`kube-system`** | Hosts system-critical components like control plane services, DNS, and networking plugins.    | Yes.                      |

 

### **Why These Namespaces Exist**
Namespaces help organize and isolate Kubernetes resources. Here's why these specific namespaces are necessary:
1. **`kube-flannel`**: Dedicated to Flannel network plugin to keep networking components separate.
2. **`node-lease`**: Improves performance by reducing frequent API calls for node health checks.
3. **`kube-public`**: Allows sharing non-sensitive cluster information publicly.
4. **`kube-system`**: Keeps critical system components isolated from user workloads.


View Namespaces and Pods
```shell
kubectl get namespaces
NAME                 STATUS   AGE
default              Active   27d
kube-node-lease      Active   27d
kube-public          Active   27d
kube-system          Active   27d
local-path-storage   Active   27d
controlplane:~$ kubectl get pods -n default
No resources found in default namespace.
controlplane:~$ kubectl get pods -n kube-public
No resources found in kube-public namespace.
controlplane:~$ kubectl get pods -n kube-node-lease
No resources found in kube-node-lease namespace.
controlplane:~$ kubectl get pods -n local-path-storage
NAME                                      READY   STATUS    RESTARTS      AGE
local-path-provisioner-76f88ddd78-srw9b   1/1     Running   2 (23m ago)   27d
```

Kube System Namespace
```shell
controlplane:~$ kubectl get pods -n kube-system
NAME                                      READY   STATUS    RESTARTS      AGE
calico-kube-controllers-7bb4b4d4d-5x7wv   1/1     Running   2 (20m ago)   27d
canal-hwn8t                               2/2     Running   2 (20m ago)   27d
canal-kkbdg                               2/2     Running   2 (20m ago)   27d
coredns-76bb9b6fb5-6hd7r                  1/1     Running   1 (20m ago)   27d
coredns-76bb9b6fb5-zsjp6                  1/1     Running   1 (20m ago)   27d
etcd-controlplane                         1/1     Running   3 (20m ago)   27d
kube-apiserver-controlplane               1/1     Running   3 (20m ago)   27d
kube-controller-manager-controlplane      1/1     Running   2 (20m ago)   27d
kube-proxy-cf4m9                          1/1     Running   1 (20m ago)   27d
kube-proxy-k5v4v                          1/1     Running   2 (20m ago)   27d
kube-scheduler-controlplane               1/1     Running   2 (20m ago)   27d
```
Config map from kube public namespace
```shell
controlplane:~$ kubectl get configmap -n kube-public
NAME               DATA   AGE
cluster-info       2      27d
kube-root-ca.crt   1      27d
```
Creating Namespace
```shell
controlplane:~/myhome$ vi namespaces.yml
controlplane:~/myhome$ kubectl apply -f namespaces.yml 
namespace/dev-environment created
controlplane:~/myhome$ kubectl get namespaces
NAME                 STATUS   AGE
default              Active   27d
dev-environment      Active   13s
kube-node-lease      Active   27d
kube-public          Active   27d
kube-system          Active   27d
local-path-storage   Active   27d
```

 
When a Kubernetes resource is **namespaced**, it means:
- It belongs to a specific **namespace**.
- It is **isolated** from resources in other namespaces.
- It **cannot interact** with resources outside its namespace unless explicitly configured.

By default, Kubernetes comes with **four namespaces**:
- **default** → The default namespace for resources.
- **kube-system** → Reserved for system resources like CoreDNS.
- **kube-public** → Publicly readable resources.
- **kube-node-lease** → Manages node heartbeats.

  
> Some resources **must** exist within a namespace, while others are **cluster-wide**.

| **Namespaced Resources** | **Non-Namespaced Resources** Cluster wide  |
|----------------------|----------------------|
| Pods, Deployments, Services, Jobs ,CronJobs | Nodes |
| ConfigMaps, Secrets | PersistentVolumes |
| Role (RBAC), ServiceAccounts | ClusterRole (RBAC) |
| NetworkPolicies | Cluster-wide StorageClasses |

 
### **Creating & Using Namespaces**
 
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: dev-environment
```
Apply it:
```sh
kubectl apply -f namespace.yaml
```

 
 **Deploy a Pod in a Specific Namespace**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  namespace: dev-environment
spec:
  containers:
    - name: nginx
      image: nginx
```
 

 **List Resources in a Namespace**
```sh
kubectl get pods --namespace=dev-environment
```

or set a default namespace:
```sh
kubectl config set-context --current --namespace=dev-environment
kubectl get pods  # No need to specify --namespace now
```

 ```shell
controlplane:~/myhome$ kubectl apply -f mypod.yml -n dev-environment
pod/my-pod created
controlplane:~/myhome$ kubectl get pods -n dev-environment   
NAME     READY   STATUS    RESTARTS   AGE
my-pod   1/1     Running   0          21s
```

## **Communication Between Namespaces**
By default, namespaces **isolate** resources. To communicate across namespaces, you need:
- **Explicit Service discovery** (e.g., `service-name.namespace.svc.cluster.local`)
- **NetworkPolicies** to allow traffic
- **RBAC rules** for access control

Example:  
A pod in `namespace-A` can access a service in `namespace-B` using:
```sh
curl http://my-service.namespace-B.svc.cluster.local
```
 Prerequisites

* Istio installed in the cluster
* **Sidecar injection enabled** in both namespaces

```bash
kubectl label namespace ns-a istio-injection=enabled
kubectl label namespace ns-b istio-injection=enabled
```

  How Istio discovers services across namespaces

Istio uses **Kubernetes DNS** + **Envoy sidecars**.

Service FQDN format:

```
<service>.<namespace>.svc.cluster.local
```

Example:

```
orders.ns-a.svc.cluster.local
```

You **must use FQDN** for cross-namespace calls.
  Basic cross-namespace call (no policy)

From a pod in `ns-b`:

```bash
curl http://orders.ns-a.svc.cluster.local:8080
```

If no Istio security policies exist, traffic flows normally.

 Enable mTLS (recommended)

 PeerAuthentication (namespace or mesh wide)

```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: ns-a-mtls
  namespace: ns-a
spec:
  mtls:
    mode: STRICT
```

This enforces **encrypted service-to-service traffic**.

 Allow cross-namespace access (AuthorizationPolicy)

By default, **STRICT mTLS blocks traffic unless allowed**.

### Allow `ns-b` → service in `ns-a`

```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-ns-b
  namespace: ns-a
spec:
  rules:
  - from:
    - source:
        namespaces: ["ns-b"]
```

 Restrict to a specific service (best practice)

```yaml
spec:
  selector:
    matchLabels:
      app: orders
  rules:
  - from:
    - source:
        namespaces: ["ns-b"]
```

Only `orders` service accepts traffic from `ns-b`.
 How identity works in Istio

Istio assigns identities like:

```
spiffe://cluster.local/ns/ns-b/sa/default
```

You can restrict access using **service accounts**:

```yaml
source:
  principals:
  - "cluster.local/ns/ns-b/sa/payments-sa"
```

  Traffic management still works across namespaces

You can use:

* `VirtualService`
* `DestinationRule`
* Retries, timeouts, circuit breakers
 

```yaml
host: orders.ns-a.svc.cluster.local
```

 
 

  **Deleting a Namespace**
**⚠️ This will delete all resources inside the namespace!**
```sh
kubectl delete namespace dev-environment
```
 
 

### **Role-Based Access Control (RBAC) in Namespaces**
RBAC in Kubernetes allows fine-grained permissions by **assigning roles to users, groups, or service accounts**.

> **Example: Create a Role & Bind It to a User**

**Step 1: Create a Role in a Namespace**
This role allows users to manage pods but not other resources.

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
Apply it:
```sh
kubectl apply -f role.yaml
```

 **Step 2: Bind the Role to a User**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: user-binding
  namespace: dev-environment
subjects:
  - kind: User
    name: developer1  # Change this to your actual username
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-manager
  apiGroup: rbac.authorization.k8s.io
```
Apply it:
```sh
kubectl apply -f rolebinding.yaml
```
  Now, `developer1` can **manage pods** in `dev-environment` but cannot create services or modify other resources.
 

### **Resource Quotas in Namespaces**
To prevent one namespace from consuming too many resources, we use **ResourceQuotas**.

### **Example: Set CPU, Memory, and Pod Limits**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: dev-quota
  namespace: dev-environment
spec:
  hard:
    pods: "10"
    requests.cpu: "4"
    requests.memory: "8Gi"
    limits.cpu: "10"
    limits.memory: "16Gi"
```
Apply it:
```sh
kubectl apply -f resource-quota.yaml
```
  This ensures that:
- Maximum **10 pods** can run in `dev-environment`
- **CPU requests** cannot exceed **4 vCPUs**
- **Memory requests** cannot exceed **8GiB**
- **CPU & memory limits** ensure fair usage

To check quota usage:
```sh
kubectl get resourcequota dev-quota --namespace=dev-environment
```

 
By default, all namespaces **can** communicate unless a **NetworkPolicy** is enforced.

 > **Deny All Incoming Traffic in `dev-environment`**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
  namespace: dev-environment
spec:
  podSelector: {}
  policyTypes:
    - Ingress
```
  This **blocks** all incoming traffic to any pod in `dev-environment`.
  
> **Allow Traffic from a Specific Namespace (`staging-environment`)**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-staging
  namespace: dev-environment
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: staging-environment
```
  This allows **only pods from `staging-environment`** to access `dev-environment`.

> Testing Cross-Namespace Communication**
 **Create a service in `dev-environment`**
```sh
kubectl run nginx --image=nginx --namespace=dev-environment
kubectl expose pod nginx --port=80 --target-port=80 --namespace=dev-environment
```

  **Try accessing it from `staging-environment`**
```sh
kubectl run test-client --image=busybox --namespace=staging-environment --rm -it -- /bin/sh
wget -O- nginx.dev-environment.svc.cluster.local
```
  If `NetworkPolicy` allows it, this request will succeed. Otherwise, it will fail.

 

# **5️⃣ Summary Table**
| **Concept** | **Description** | **Example YAML** |
|------------|---------------|---------------|
| **RBAC Role** | Restrict users to specific actions | `role.yaml` |
| **RoleBinding** | Assign role to a user/service | `rolebinding.yaml` |
| **Resource Quota** | Limit CPU, memory, pods per namespace | `resource-quota.yaml` |
| **NetworkPolicy (Deny All)** | Block all ingress traffic | `deny-all.yaml` |
| **NetworkPolicy (Allow Specific Namespace)** | Allow traffic from `staging-environment` | `allow-staging.yaml` |
 
