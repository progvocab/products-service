A **DaemonSet** in Kubernetes is a workload resource that ensures **a copy of a specific Pod runs on every node** (or a selected set of nodes) in the cluster.

## 🔧 **What Does a DaemonSet Do?**

A DaemonSet automatically deploys and manages Pods so that **each node always has exactly one instance** of that Pod running.
If new nodes join, Kubernetes adds the DaemonSet Pod to them.
If nodes leave, the Pod is removed from those nodes.

## 📌 **Common Use Cases**

DaemonSets are typically used for node-level operations:

### 1. **Log Collection**

* Tools like **Fluentd**, **Filebeat**, **Logstash** run on every node to collect logs.

### 2. **Monitoring Agents**

* Tools like **Prometheus Node Exporter**, **Datadog Agent**.

### 3. **Networking Components**

* CNI plugins like **Calico**, **Weave**, **Flannel** use DaemonSets to run networking agents on all nodes.

### 4. **Storage Daemons**

* GlusterFS, Ceph daemons, etc.

---

## 📘 **Key Properties**

| Property                      | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| Ensures one Pod per node      | Automatically schedules a Pod on *every* node.      |
| Supports node selectors       | You can limit the DaemonSet to certain node groups. |
| Handles node addition/removal | Automatically manages Pods as nodes join/leave.     |
| Uses its own controller       | Not part of Deployment/ReplicaSet.                  |

---

## 🧩 **Simple DaemonSet YAML**

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      containers:
      - name: node-exporter
        image: prom/node-exporter
        ports:
        - containerPort: 9100
```

---

## 🧠 **How is DaemonSet different from Deployment?**

| Feature                                | DaemonSet  | Deployment           |
| -------------------------------------- | ---------- | -------------------- |
| Pods per node                          | 1 per node | Any number, anywhere |
| Node-level tasks                       | Yes        | No                   |
| Automatically add Pods when node joins | Yes        | No                   |
| Rolling updates                        | Yes        | Yes                  |

---

## 🎯 **In simple words**

A **DaemonSet = Run this Pod on every node**.

If you want a component that should exist on **each node**, use a DaemonSet.

---

If you want, I can also explain:
✔ DaemonSet with tolerations
✔ DaemonSet with nodeSelectors / nodeAffinity
✔ DaemonSet vs StatefulSet vs Deployment with real-world examples
