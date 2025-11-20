


The Kubernetes kube-scheduler assigns Pods to nodes through a two-phase process: filtering and scoring. 
In the filtering phase, the scheduler removes all nodes that cannot host the Pod based on resource requests (CPU/memory), node conditions (Ready, DiskPressure), taints and tolerations, affinity/anti-affinity rules, and topology constraints. 

After filtering, the scoring phase evaluates the remaining nodes using scoring plugins—such as resource balance, spreading, and node affinity—to assign a weighted score to each node. The scheduler then selects the node with the highest score and binds the Pod to it, ensuring an optimal placement based on health, capacity, and policy constraints.


A Deployment manages stateless workloads by creating a desired number of identical Pods and allowing Kubernetes to schedule them on any suitable node. It supports strategies like rolling updates and ensures replicas can scale up or down as needed. 

A DaemonSet, by contrast, ensures exactly one Pod runs on every node (or on specific nodes using node selectors or tolerations), making it ideal for system-level agents like log collectors or network plugins. Unlike Deployments, DaemonSets focus on node coverage rather than replica count, and Pods are created automatically on node join events, ensuring continuous presence across the cluster.


A Kubernetes StatefulSet maintains stable Pod identity and storage by assigning each Pod a fixed ordinal index, which becomes part of its persistent DNS name (such as pod-0, pod-1). This stable naming ensures ordered startup and shutdown, allowing distributed systems to maintain membership consistency. 


Each Pod is also associated with its own dedicated PersistentVolumeClaim (PVC), ensuring that storage remains attached even if the Pod is rescheduled to a different node. Together, the deterministic identity, stable network address, and persistent storage allow StatefulSets to run stateful applications like databases and quorum-based systems reliably.




Pods communicate directly using Pod IPs, which are routable across the entire cluster. Kubernetes requires that every Pod can reach every other Pod without NAT, even if they live on different nodes. This cross-node routing is provided by the CNI plugin (like Calico, Flannel, Cilium), which configures node-level routes and network interfaces. kube-proxy programs iptables or IPVS rules only for Service IPs, enabling stable virtual IPs for Services and load balancing traffic to Pods, not between Pods. So Pod-to-Pod communication uses Pod IPs through CNI networking, while Service IPs are used only when accessing a Service abstraction, not for direct Pod communication.


A readiness probe determines when a container is ready to receive traffic; only after it succeeds does the Pod get added to the Service endpoints so traffic can reach it. 

A liveness probe is a continuous health check—if it fails, kubelet restarts the container, and repeated failures lead to CrashLoopBackOff.

 A startup probe is used for applications that take a long time to initialize; during this period, Kubernetes ignores liveness and readiness failures. Once the startup probe succeeds, Kubernetes begins evaluating liveness and readiness normally. This prevents slow-starting applications from being killed prematurely by liveness checks.


ConfigMaps store non-sensitive configuration data in plain text and are not encrypted by default, meaning their values can be easily viewed. They are intended for application configuration but can be reused across multiple Deployments or services if referenced properly.


 Secrets, on the other hand, are designed for sensitive data such as passwords and tokens; they are stored in etcd in a base64-encoded form and can be encrypted at rest using Kubernetes encryption providers. Secrets can also be reused across many workloads, but kubelet treats them with stricter controls—such as not writing them to disk unless explicitly required—making them more secure than ConfigMaps.



The Horizontal Pod Autoscaler (HPA) continuously checks Pod metrics—such as CPU, memory, or custom application metrics—collected by the Metrics Server or external metric providers. Based on the target thresholds defined in the HPA, Kubernetes compares the current utilization with the desired utilization and computes the required number of replicas using a control loop formula. If the observed metric exceeds the threshold, HPA increases the number of Pods; if it drops below, HPA scales them down. This ensures that the Deployment automatically matches workload demand by adjusting replicas in real time.

Here is the exact HPA control loop formula explained clearly and concisely:

HPA Control Loop Formula

Kubernetes calculates the desired number of replicas using:

\text{Desired Replicas} = \text{Current Replicas} \times \frac{\text{Current Metric Value}}{\text{Target Metric Value}}

What this means

Current Metric Value = actual CPU/memory/custom metric observed per Pod

Target Metric Value = value you defined in the HPA (ex: target CPU = 50%)

If the current metric is higher than the target → scale up

If the current metric is lower than the target → scale down


Example

Assume:

Current replicas = 5

Target CPU = 50%

Observed average CPU = 80%


\text{Desired Replicas} = 5 \times \frac{80}{50} = 5 \times 1.6 = 8

So HPA increases Pods from 5 → 8.

Another Example (scale down)

Current replicas = 10

Target CPU = 60%

Observed CPU = 30%


\text{Desired Replicas} = 10 \times \frac{30}{60} = 10 \times 0.5 = 5

HPA scales down from 10 → 5.

In one line

HPA scales replicas proportionally based on how far the current metric deviates from the target metric, using a continuous feedback loop.


A PersistentVolume (PV) is a cluster-level storage resource provisioned by an administrator or through dynamic provisioning using a StorageClass, and it represents the actual physical or cloud-backed storage. 



A PersistentVolumeClaim (PVC) is a user request for storage that specifies requirements such as size and access mode. When a PVC is created, Kubernetes finds a matching PV (or dynamically provisions one) and binds them together in a one-to-one relationship. The Pod then mounts the PVC, and Kubernetes ensures the underlying PV remains attached even if the Pod is rescheduled, providing durable and portable storage.



The kubelet is the node-level agent responsible for ensuring that all Pods assigned to its node are running and healthy. It constantly communicates with the API server to receive Pod specs and report Pod status, and it uses the container runtime (such as containerd or CRI-O) to create, start, stop, and delete containers. The kubelet also executes liveness, readiness, and startup probes, and restarts containers when probes fail. Alongside this, it integrates with cAdvisor to collect CPU, memory, and filesystem metrics for each container. Through these actions, the kubelet manages the full lifecycle of Pods on a node and keeps the local state aligned with the desired state defined by the control plane.