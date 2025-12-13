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



If you want, I can also break down who pays in a multi-account or multi-tenant EKS setup, or compare this with GKE / AKS pricing.