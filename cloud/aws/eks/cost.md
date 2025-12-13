In Amazon EKS, the control plane cost is paid by you (the AWS account that creates the cluster) — but it’s managed and operated by AWS.

How EKS control plane pricing works

AWS runs and manages the Kubernetes control plane (API server, etcd, controller manager, scheduler).

You are charged a fixed fee per EKS cluster, regardless of how many nodes or pods you run.


Current pricing (standard EKS)

$0.10 per hour per cluster

≈ $72 per month per cluster


This cost includes:

Highly available control plane (multi-AZ)

Automatic patching and upgrades

Secure etcd management

Kubernetes API server availability


What you do not pay for (directly)

EC2 instances for control plane (you don’t see them)

Load balancers for API server

etcd storage and replication


What you do pay for separately

Worker nodes (EC2, Fargate, or managed node groups)

EBS volumes, ALBs/NLBs

Data transfer

Add-ons (if applicable)


Special cases

EKS on Fargate → still pay the same control plane fee

EKS Auto Mode → still includes the same control plane pricing

Self-managed Kubernetes on EC2 → no control plane fee, but you manage everything


Simple comparison

Kubernetes option	Control plane cost

EKS (managed)	You pay $0.10/hr
EKS Anywhere	You pay infra only
Self-managed k8s on EC2	No fixed fee
OpenShift / other managed	Varies


Key takeaway

> AWS owns and operates the EKS control plane, but you pay for it as a managed service.




## Worker Node Cluster Cost

In **Amazon EKS**, **there is NO separate “EKS worker node cluster” charge**.
You are charged **only for the AWS infrastructure that runs the worker nodes**, not for EKS itself.

> *“EKS charges only for the control plane at $0.10 per hour; worker nodes are billed separately as standard EC2, EBS, networking, and load balancer resources.”*
> “Worker nodes are charged as normal EC2/EBS/Network resources.”
 

### Charged by EKS

### EKS Control Plane

* **$0.10 per hour per cluster**
* Charged regardless of worker nodes
* Covers:

  * API server
  * etcd
  * Control plane HA (multi-AZ)

  This is **independent of worker nodes**

 

### Not charged by EKS

* **Worker nodes**
* **Pods**
* **Kubernetes scheduling**
* **Cluster Autoscaler / HPA**

  Worker nodes are billed as **normal AWS compute**

 

### What you actually pay for worker nodes

### Case 1: EC2 Worker Nodes (Most Common)

You pay for:

### a) EC2 instances

* Based on:

  * Instance type (t3.medium, m6i.large, etc.)
  * On-Demand / Spot / Reserved
* Charged per second (Linux)

### b) EBS volumes

* Root volume for each node
* Any attached persistent volumes

### c) Networking

* Data transfer (inter-AZ, internet egress)
* NAT Gateway (if nodes are in private subnets)

### d) Load Balancers

* ALB / NLB created by Services or Ingress

 

```
3 × t3.medium EC2 nodes
+ 3 × 30GB EBS
+ ALB
= Regular EC2 + EBS + ALB pricing
```

 

### Case 2: Managed Node Groups

* **No extra charge**
* Just a convenience layer
* Same billing as EC2 instances

  Managed ≠ More expensive

 

### Case 3: Self-Managed Node Groups

* Same cost as EC2
* You manage upgrades and lifecycle

 

### Case 4: Fargate (Serverless Pods)

* **No worker nodes**
* You pay per:

  * vCPU seconds
  * Memory seconds
* More expensive than EC2 for steady workloads
* Good for spiky / low-ops workloads

 
 

| Component          | Charged By | Cost           |
| ------------------ | ---------- | -------------- |
| EKS Control Plane  | EKS        | $0.10/hr       |
| EC2 Worker Nodes   | EC2        | Instance price |
| Managed Node Group | EC2        | No extra       |
| Self-Managed Nodes | EC2        | No extra       |
| Fargate Pods       | Fargate    | vCPU + memory  |
| EBS Volumes        | EBS        | GB/month       |
| Load Balancers     | ELB        | Hour + LCU     |
| NAT Gateway        | VPC        | Hour + data    |

 
 
 



 

### Example monthly cost (simple)

Assume:

* 1 EKS cluster
* 3 × t3.medium nodes (on-demand)
* 30 GB EBS each
* 1 ALB

Rough breakdown:

* EKS control plane → ~$73/month
* EC2 nodes → EC2 pricing
* EBS → storage pricing
* ALB + data transfer → usage-based

  **Most of the cost is worker nodes, not EKS**

 

### Cost optimization tips for worker nodes

* Use **Spot instances** for stateless workloads
* Use **Cluster Autoscaler** to scale nodes to zero
* Use **multiple node groups** (on-demand + spot)
* Right-size instance types
* Move bursty workloads to **Fargate**
* Use **Savings Plans / Reserved Instances**

 
 
 

 


 
 

 

##    EKS Fargate pricing  


Even with **EKS + Fargate**, **you may still be charged even if no API calls hit your pod**, depending on whether the pod is running.

###   What you pay for with Fargate

For **each running pod on Fargate**, you pay for:

* **vCPU-seconds**
* **Memory-seconds**

- **Billing starts when the pod starts**
- **Billing stops when the pod is terminated**

> Traffic to the pod does **not** matter.

 

###   Case 1: Pod is NOT running

Examples:

* Deployment scaled to **0 replicas**
* Job completed and pod terminated

 - **No Fargate compute charges**
 - **Only EKS control plane cost applies** ($0.10/hr)

  This is the only “no charge” case

 

###   Case 2: Pod is running but idle

 

* Deployment has 1 replica
* Pod is waiting for traffic
* No API calls

  **You are still billed**

* vCPU-seconds
* Memory-seconds

  Fargate does **not** scale to zero automatically

 

###   Case 3: Pod is running but blocked (sleep / waiting / polling)

Even if:

* Thread is sleeping
* App is idle
* No requests

  Billing continues while pod exists

 

> “Serverless = no traffic = no cost”

  **True for AWS Lambda**
  **False for Fargate**

Fargate is **serverless infrastructure**, not **event-driven compute**.
 

### Comparison: Lambda vs Fargate

| Feature                 | Lambda                | EKS Fargate |
| ----------------------- | --------------------- | ----------- |
| Scales to zero          | ✅                     | ❌           |
| Billed only on requests | ✅                     | ❌           |
| Billing unit            | Invocation + duration | Pod runtime |
| Idle cost               | $0                    | > $0        |
| Long-running workloads  | ❌                     | ✅           |

 
### When would Fargate still be “zero-ish”?

Only if:

* Pods are **created on demand** (Jobs, CronJobs)
* Pods **exit immediately after work**
* Replica count is **0 most of the time**

Example:

* CronJob runs once per hour
* Pod runs for 2 minutes
* You pay only for those 2 minutes

 

### EKS Control Plane cost still applies

Even if:

* No pods
* No traffic

You still pay:

```
$0.10/hour per EKS cluster
```

 

### Best practices to avoid idle Fargate cost

###   Scale deployments to zero

* Use **KEDA**
* Use **HPA with external metrics**
* Or trigger pods via Jobs

 

###   Prefer Jobs / Event-driven pods

Good for:

* Batch processing
* Event handlers
* Scheduled tasks

 

###  Use Lambda if true zero-idle cost is required

If:

* Request-driven
* Short execution
* Stateless

 

### Real-world summary

> “With EKS Fargate, you pay for pods while they are running, even if they receive no traffic. You only stop paying when the pod is terminated.”
 

 

> *“EKS Fargate charges for pod runtime, not traffic; a running but idle pod still incurs cost, and only terminated pods stop billing.”*

More : 

* Compare **Lambda vs Fargate vs EC2** for your workload
* Show how to **auto-scale Fargate pods to zero**
* Estimate **real monthly cost** for your current setup
* Suggest a **hybrid EKS architecture** to minimize idle spend
* break down who pays in a multi-account or multi-tenant EKS setup, or compare this with GKE / AKS pricing.
*  Compare **EKS vs ECS vs Lambda cost**
* Calculate **exact monthly cost** for your node types
* Explain **why EKS looks expensive but usually isn’t**
* Design a **low-cost EKS architecture**
