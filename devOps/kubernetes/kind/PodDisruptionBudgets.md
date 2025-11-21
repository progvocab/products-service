### What Is a Pod Disruption Budget (PDB)

A **Pod Disruption Budget (PDB)** in Kubernetes defines the *minimum number* or *minimum percentage* of Pods that must remain available during **voluntary disruptions**.
Voluntary disruptions include actions by **kube-controller-manager**, such as draining a node (e.g., during cluster upgrade, node replacement).
PDB ensures services remain available by preventing evictions that would drop the number of replicas below the budget.

### YAML Example: PodDisruptionBudget

This PDB ensures **at least 2 Pods must remain available** for the `orders-service` Deployment.

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: orders-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: orders-service
```

### Related Kind: Deployment YAML That Works With PDB

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders-service
spec:
  replicas: 4
  selector:
    matchLabels:
      app: orders-service
  template:
    metadata:
      labels:
        app: orders-service
    spec:
      containers:
        - name: orders-app
          image: myregistry/orders:1.0
```

### How They Work Together

* **kubelet** does not handle PDB.
* **kube-controller-manager** enforces the PDB during evictions.
* When someone drains a node (`kubectl drain`), eviction requests are sent to the **Eviction API**.
* The Eviction API checks the PDB and denies eviction if it would violate `minAvailable` or `maxUnavailable`.

If you want, I can also share:

* A PDB using `maxUnavailable`
* A mermaid diagram showing the eviction vs. PDB decision flow
