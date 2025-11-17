### Medium Level Kubernetes Question 1

Your application has intermittent latency spikes in a Kubernetes cluster. How would you investigate whether the issue is related to pod resource limits, node pressure, network policies, or kube-proxy configuration?

### Medium Level Kubernetes Question 2

A Deployment is stuck with pods in CrashLoopBackOff. What steps would you take using logs, events, readiness probes, and ConfigMap or Secret debugging to isolate the cause?

### Medium Level Kubernetes Question 3

Explain how you would design a zero-downtime rolling update strategy for a production microservice. Include how you would configure readiness probes, maxSurge, maxUnavailable, and PodDisruptionBudgets.

### Medium Level Kubernetes Question 4

A service exposed via an Ingress is reachable internally but not externally. What checks would you perform involving Ingress controller logs, service type, DNS propagation, and security groups or firewalls?

### Medium Level Kubernetes Question 5

Your StatefulSet pods are failing to mount a PersistentVolumeClaim. How would you debug issues related to storage classes, dynamic provisioning, reclaim policies, and node-availability constraints?

If you want, I can also answer these or generate harder scenario-based questions.


### Answer to Question 1

When an application shows intermittent latency spikes in Kubernetes, start by checking whether the pods are CPU or memory throttled using

```
kubectl top pod
```

and compare usage with resource requests and limits. Inspect node pressure conditions using

```
kubectl describe node <node>
```

to detect disk, memory, or PID pressure. If the problem might be networking, review NetworkPolicy rules, CNI plugin logs, and verify whether kube-proxy is using iptables or IPVS and if rules are correctly programmed. Finally, check pod-level logs, service latency metrics, and recent deployments to determine whether the issue is inside the application or caused by cluster-level resource contention.


### Liveness Probe

A liveness probe checks whether the application inside a pod is **alive** and functioning. If this probe fails, Kubernetes **kills and restarts** the container. It is used to recover from deadlocks, hangs, or situations where the app is running but not progressing.

### Readiness Probe

A readiness probe checks whether the application is **ready to receive traffic**. If it fails, Kubernetes **removes the pod from service endpoints** but does not restart it. This ensures that only fully initialized and healthy pods serve requests.

### Key Differences

* Liveness failure restarts the container; readiness failure only removes it from load balancing.
* Liveness detects dead apps; readiness detects apps not yet ready or temporarily unable to serve traffic.
* Liveness maintains application health; readiness maintains service availability.

### Example of Readiness and Liveness in a Pod Spec

```
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 3
```

This configuration restarts the pod if `/health/live` fails but only stops traffic flow if `/health/ready` fails.


### maxSurge

`maxSurge` controls how many **extra pods** can be created during a rolling update beyond the desired replica count. It helps achieve zero-downtime deployments by temporarily adding capacity so that new pods can start before old ones terminate. A value like `maxSurge: 1` means one additional pod can run during updates. This ensures smooth transitions when applications require warm-up time or slow startup.

### maxUnavailable

`maxUnavailable` defines how many pods can be **unavailable at the same time** during a rolling update. Setting `maxUnavailable: 0` ensures that Kubernetes never removes a running pod until a new one is fully ready, enabling strict high availability. Higher values allow faster updates but introduce the risk of reduced service capacity. Choosing the right value depends on traffic patterns and tolerance for temporary downtime.

### PodDisruptionBudget

A PodDisruptionBudget (PDB) specifies how many pods in a replica set or deployment must remain **available during voluntary disruptions** such as node drains or upgrades. It prevents Kubernetes from evicting too many pods simultaneously, ensuring service stability during maintenance events. PDBs can be defined using `minAvailable` or `maxUnavailable`. They are essential for stateful or high-availability workloads that cannot afford multiple simultaneous pod losses.


### Answer to Question 4

When an Ingress is reachable internally but not externally, start by checking whether the **Ingress Controller** (such as NGINX Ingress Controller or AWS ALB Controller) has correctly created or updated the external load balancer. Next, verify that the **Service of type NodePort or LoadBalancer**, created by the controller, is exposing the correct node ports and that the **kube-proxy** on each node has programmed the iptables or IPVS rules needed for traffic routing. Inspect the **Cloud Provider APIs** (for example, AWS ELB, GCP Load Balancer, Azure Front Door) to ensure the public IP, listener rules, and security groups allow inbound traffic. Finally, confirm that **CoreDNS** is resolving the Ingress hostname internally and that external DNS records (managed by Route53, Cloud DNS, or others) correctly map to the load balancer’s public IP so that traffic can reach the cluster’s edge.


### Answer to Question 5 

When StatefulSet pods fail to mount a PersistentVolumeClaim, first check whether the **StorageClass** (provisioned by a CSI driver like EBS CSI, Ceph CSI, or NFS CSI) is correctly configured and supports dynamic provisioning. Kubernetes will then ask the **CSI Provisioner** to create a PV; if provisioning fails, the CSI controller logs show the exact error. Next, inspect the **kubelet** on the target node, because it is responsible for attaching and mounting the volume; issues like node not having required IAM permissions, missing drivers, or mount failures appear in kubelet logs. Finally, confirm that **scheduler** constraints such as node affinity, topology zones, or underlying cloud provider storage availability match the PVC request; mismatches cause pods to remain Pending or repeatedly fail volume attachment.

