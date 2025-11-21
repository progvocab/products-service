### Node Selector

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

---

### Node Affinity

Node Affinity is a **more advanced and flexible replacement** for `nodeSelector`.
It lets you define **rules with operators, soft/hard preferences, and ranges**.
The **kube-scheduler** interprets these rules during scheduling.

---

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

---

### Node Selector vs Node Affinity (Concise)

| Feature     | Node Selector                   | Node Affinity                           |
| ----------- | ------------------------------- | --------------------------------------- |
| Flexibility | Low                             | High                                    |
| Operators   | Only exact match                | In, NotIn, Exists, DoesNotExist, Gt, Lt |
| Priority    | Hard only                       | Hard + Soft                             |
| Recommended | Deprecated in favor of Affinity | Preferred                               |

---

### Components Involved

| Step           | Component                    | Responsibility                    |
| -------------- | ---------------------------- | --------------------------------- |
| Pod submission | **kubectl → kube-apiserver** | Pod spec stored                   |
| Scheduling     | **kube-scheduler**           | Evaluates labels / affinity rules |
| Placement      | **kubelet**                  | Runs Pod on chosen node           |

---

If you like, I can also generate a **mermaid diagram** showing how scheduler evaluates affinity or compare **nodeAffinity vs podAffinity vs tolerations**.
