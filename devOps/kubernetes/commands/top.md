### What Is `kubectl top` Command

`kubectl top` displays **real-time CPU and memory usage** of Kubernetes nodes and Pods. It fetches metrics from the **Metrics Server**, not from the kubelet directly. The Metrics Server aggregates resource usage reported by **kubelet’s cAdvisor** and exposes it through the **Resource Metrics API**.
`kubectl top` is mainly used for debugging performance issues, checking Pod pressure, validating autoscaler behavior, and monitoring cluster health.

### What `kubectl top` Does Internally

| Step | Component      | Description                                                        |
| ---- | -------------- | ------------------------------------------------------------------ |
| 1    | kubectl        | Sends a request to the Metrics API (`apis/metrics.k8s.io/v1beta1`) |
| 2    | kube-apiserver | Routes request to Metrics Server                                   |
| 3    | Metrics Server | Fetches fresh stats from kubelet/cAdvisor                          |
| 4    | kubelet        | Reads CPU/Memory usage from cAdvisor and exposes it                |
| 5    | Metrics Server | Aggregates and returns usage metrics                               |
| 6    | kubectl        | Displays them to the user                                          |

### Pre-requisite

Metrics Server must be installed.
Without it, you get:
`error: Metrics API not available`

### Basic Usage

#### Show resource usage for all nodes

```bash
kubectl top nodes
```

#### Example Output

```
NAME        CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
node-1      450m         30%    2.5Gi           60%
node-2      300m         18%    1.8Gi           40%
```

#### Show usage for all Pods in a namespace

```bash
kubectl top pods -n orders
```

#### Show usage for a specific Pod

```bash
kubectl top pod orders-app-7df9f4974d-9g2kw -n orders
```

### What It Is Commonly Used For

* Checking Pods with high CPU/memory usage
* Troubleshooting OOMKills
* Validating HorizontalPodAutoscaler scaling behavior
* Identifying overloaded nodes
* Debugging performance issues before a deployment

If you want, I can also provide:

* A mermaid diagram of the internal metric flow
* Examples of HPA decisions using values from `kubectl top`
