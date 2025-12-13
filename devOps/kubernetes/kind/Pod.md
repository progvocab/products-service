# POD

In Kubernetes, **Pods** are the smallest deployable unit, but they can exist in **different “types” depending on how they are created, managed, and used**. 

*“Kubernetes Pods can be categorized based on lifecycle management (Deployment, StatefulSet, DaemonSet, Job), composition (single or multi-container), QoS class, networking mode, and special behaviors like init or ephemeral containers.”*
 

##   Based on Creation & Lifecycle

### 1. Static Pods

* **Created directly by kubelet**, not via API server.
* Defined as YAML files on the node (e.g. `/etc/kubernetes/manifests`).
* **Not managed by controllers**.
* If deleted, kubelet recreates them automatically.
* Commonly used for **control plane components** (etcd, kube-apiserver).

**Key point:**

> Visible via API server, but API server does not manage them.

 

### 2. ReplicaSet-managed Pods

* Created and managed by a **ReplicaSet**.
* Ensures **desired number of pod replicas**.
* Rarely used directly (Deployment uses ReplicaSet internally).

**Use case:** Ensuring a fixed number of identical pods.
 
### 3. Deployment Pods

* Managed by a **Deployment** controller.
* Supports **rolling updates, rollbacks, scaling**.
* Most common type for stateless applications.

**Lifecycle:**
Deployment → ReplicaSet → Pod
 

### 4. StatefulSet Pods

* Managed by a **StatefulSet**.
* Pods have:

  * Stable identity (`pod-0`, `pod-1`)
  * Stable DNS
  * Persistent volumes
* Created and deleted **in order**.

**Use case:** Databases, Kafka, ZooKeeper.

 

### 5. DaemonSet Pods

* One pod runs **on every node** (or selected nodes).
* Automatically added when a new node joins.

**Use case:**

* Logging agents (Fluentd)
* Monitoring agents (Prometheus node-exporter)
* CNI plugins
 

### 6. Job Pods

* Created by a **Job** controller.
* Run **once or until completion**.
* Retries on failure.

**Use case:** Batch processing, data migration, one-time tasks.

 

### 7. CronJob Pods

* Scheduled Jobs.
* Create **Job → Pod** at scheduled times.

**Use case:**
Backups, report generation, cleanup tasks.

 

##  Based on Pod Composition

### 8. Single-container Pod

* Most common.
* One container per pod.

**Use case:** Standard microservices.

 

### 9. Multi-container Pod

Multiple containers sharing:

* Same IP
* Same volumes
* Same lifecycle

#### Common patterns:

* **Sidecar**: log shippers, service mesh proxies (Istio Envoy)
* **Adapter**: transform data formats
* **Ambassador**: proxy access to external services

 

##  Based on Special Behavior

### 10. Init Container Pods

* One or more **init containers** run before main containers.
* Must complete successfully before app starts.

**Use case:**

* Database migrations
* Configuration setup
* Waiting for dependencies



### 11. Ephemeral Pods

* Short-lived pods without persistent state.
* Often created by Jobs or ad-hoc kubectl runs.

**Example:**

```bash
kubectl run test --image=busybox --rm -it
```

 

### 12. Ephemeral Containers (Debug Pods)

* Injected into **existing pods** for debugging.
* Not part of pod spec at creation.
* Cannot restart or be removed.

**Use case:** Debugging production issues.

 
##   Based on QoS (Quality of Service)

### 13. Guaranteed Pods

* CPU & memory **requests == limits** for all containers.
* Highest scheduling and eviction priority.

 
### 14. Burstable Pods

* Requests < limits.
* Default for most workloads.
 

### 15. BestEffort Pods

* No requests or limits set.
* First to be evicted under pressure.

 

##   Based on Networking & Exposure

### 16. HostNetwork Pods

* Use node’s network namespace.
* No pod IP.

**Use case:**
Network agents, CNI plugins.

 

### 17. HostPID / HostIPC Pods

* Share node PID or IPC namespace.

**Use case:**
System-level debugging or monitoring.

 

##   Based on Scheduling Constraints

### 18. Node-affinity / Anti-affinity Pods

* Scheduled based on node labels or other pods.

 

### 19. Tainted / Tolerated Pods

* Pods that can run on tainted nodes.

**Use case:**
Dedicated workloads (GPU, system pods).

 
 

| Category    | Pod Type                            | Purpose                  |
| ----------- | ----------------------------------- | ------------------------ |
| Lifecycle   | Static                              | Node-managed system pods |
| Lifecycle   | Deployment                          | Stateless apps           |
| Lifecycle   | StatefulSet                         | Stateful apps            |
| Lifecycle   | DaemonSet                           | Node-level agents        |
| Lifecycle   | Job / CronJob                       | Batch & scheduled work   |
| Composition | Single / Multi-container            | App + sidecars           |
| Special     | Init Containers                     | Pre-run setup            |
| Special     | Ephemeral Containers                | Debugging                |
| QoS         | Guaranteed / Burstable / BestEffort | Resource handling        |
| Networking  | HostNetwork                         | Node-level networking    |

 

 

 

##  Node Selector

Node Selector is the **simplest scheduling constraint** in Kubernetes. It allows you to run a Pod **only on nodes that contain specific labels**.
The **kube-scheduler** performs a strict match: if the labels don’t match, the Pod will **never** be scheduled.

#### Example

Node labeled with:

```bash
kubectl label node node-1 disk=ssd
```

Pod spec:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  nodeSelector:
    disk: ssd
```

**Effect:**
The kube-scheduler schedules the Pod only on nodes where `disk=ssd`.

 

### Node Affinity

Node Affinity is a **more advanced and flexible replacement** for `nodeSelector`.
It lets you define **rules with operators, soft/hard preferences, and ranges**.
The **kube-scheduler** interprets these rules during scheduling.
 

### Types of Node Affinity

#### 1. **RequiredDuringSchedulingIgnoredDuringExecution (Hard Rule)**

Pod is scheduled **only if** node matches the rules.
If no nodes match — Pod remains Pending.

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: disk
          operator: In
          values: ["ssd", "nvme"]
```

#### 2. **PreferredDuringSchedulingIgnoredDuringExecution (Soft Rule)**

Scheduler **prefers** matching nodes but will schedule on other nodes if necessary.

```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 80
      preference:
        matchExpressions:
        - key: region
          operator: In
          values: ["us-east-1"]
```

 
### Node Selector vs Node Affinity  

| Feature     | Node Selector                   | Node Affinity                           |
| ----------- | ------------------------------- | --------------------------------------- |
| Flexibility | Low                             | High                                    |
| Operators   | Only exact match                | In, NotIn, Exists, DoesNotExist, Gt, Lt |
| Priority    | Hard only                       | Hard + Soft                             |
| Recommended | Deprecated in favor of Affinity | Preferred                               |

 
### Components Involved

| Step           | Component                    | Responsibility                    |
| -------------- | ---------------------------- | --------------------------------- |
| Pod submission | **kubectl → kube-apiserver** | Pod spec stored                   |
| Scheduling     | **kube-scheduler**           | Evaluates labels / affinity rules |
| Placement      | **kubelet**                  | Runs Pod on chosen node           |

More :
* Ask you **advanced interview questions** on pod behavior
* Explain **pod vs container vs process isolation**
* Map **pod types to real production use cases** (EKS-focused)

 - how scheduler evaluates affinity or compare **nodeAffinity vs podAffinity vs tolerations**.
