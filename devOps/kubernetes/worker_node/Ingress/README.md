Ingress always works on worker nodes, not the control-plane node.

Here is the clean and correct explanation.


---

✅ Ingress Controller runs on worker nodes

An Ingress object is just YAML (a rule).
But the Ingress Controller is the actual component that handles traffic.

Examples of Ingress Controllers:

NGINX Ingress Controller

HAProxy Ingress

Traefik

Kong

Envoy Gateway

Istio Ingress Gateway


All of these run as Pods on worker nodes, usually via a DaemonSet or Deployment.

They do NOT run on control-plane nodes.


---

🧠 Why Ingress Controllers run on worker nodes?

Because they need to:

receive external traffic

run network listeners (80, 443)

act as reverse proxies

forward traffic to backend Pods


Control-plane nodes generally:

should not receive external traffic

host only API server, scheduler, controller-manager, etc.

are isolated (in EKS, GKE, AKS)


So Ingress does not run there.


---

🧩 How Ingress receives traffic

Ingress Controller Pod exposes:

a Service of type LoadBalancer, or

a NodePort


Traffic flow:

Internet
   ↓
Cloud Load Balancer
   ↓
Worker Node (NodePort)
   ↓
kube-proxy
   ↓
Ingress Controller Pod
   ↓
Backend Pods (via CNI)


---

📌 Special case: Bare-metal clusters

In bare-metal, you might use something like:

MetalLB

NodePort

HostPorts


Still, the Ingress Controller Pods run on worker nodes, not control-plane nodes.


---

🧠 Summary

Component	Runs On	Why

Ingress Controller	✔ Worker nodes	Handles L7 routing, receives external traffic
Ingress (YAML rule)	API Server	Just a config object
Control-plane nodes	✖ Not for Ingress	Reserved for K8s control components



---

If you want, I can show you:

👉 Where exactly the Ingress Pod runs
👉 Traffic flow diagrams for NodePort vs LoadBalancer
👉 Ingress vs Gateway API
👉 Ingress vs Istio Gateway

Just tell me!

Ingress and kube-proxy do not directly talk to each other, but they work together in the traffic path.
Here is the clean, accurate explanation.


---

✅ Key Idea

Ingress Controller sends and receives traffic using Kubernetes Services.
kube-proxy implements those Services.
Therefore, the interaction is indirect, through Services → Endpoints → Pod IPs.


---

🧩 Where they sit in the network

kube-proxy (L4 rules: ClusterIP, NodePort)
                      ↑
Ingress Controller (L7 routing proxy)
                      ↑
          CNI (Pod networking: PodIP, routing)

So:

Ingress Controller: L7 HTTP/HTTPS router

kube-proxy: L4 service load-balancer

CNI: L3 Pod networking



---

🧠 How Ingress interacts with kube-proxy (indirectly)

✔ 1. Ingress Controller is exposed using a Service

Almost always:

Ingress Controller → Service (NodePort or LoadBalancer)

That Service is implemented by kube-proxy on every worker node.

So external traffic hits:

External LB → NodePort (kube-proxy) → Ingress Controller Pod

How:

kube-proxy installs iptables/IPVS rules

NodePort traffic arrives on the node

kube-proxy forwards it to the Ingress Controller Pod IP


This is the first indirect interaction.


---

✔ 2. Ingress Controller forwards traffic to backend Services

Ingress routes something like:

/api → serviceA
/web → serviceB

Ingress Controller does:

1. Reads the Endpoints of the backend Services


2. Gets Pod IPs (endpoints) from API server


3. Sends traffic to those Pod IPs



But kube-proxy also installs rules for these Services:

ClusterIP → Pod IPs (load balancing)

Thus backend traffic flows:

Ingress Controller Pod
     ↓
ServiceA (ClusterIP)
     ↓ (via kube-proxy)
Pod IPs (via CNI)

This is the second indirect interaction.


---

🧩 Full Traffic Path (complete picture)

Internet
   ↓
Cloud LB (AWS/GCP/Azure)
   ↓
Worker Node
   ↓
kube-proxy (NodePort rules)
   ↓
Ingress Controller Pod (via CNI)
   ↓
Ingress rules (/api → serviceA)
   ↓
ClusterIP Service (via kube-proxy rules)
   ↓
Backend Pods (via CNI routing)

kube-proxy is involved twice:

1. Exposing the Ingress Controller Pod


2. Forwarding requests to backend services




---

📌 Summary Table

Component	Layer	Responsibility	Interaction with Ingress

Ingress Controller	L7	HTTP routing	Uses kube-proxy rules to receive traffic (NodePort)
kube-proxy	L4	Implements Services (NodePort, ClusterIP)	Forwards traffic to Ingress Controller & then to backends
CNI	L3	Pod networking	Carries Pod-to-Pod traffic



---

📌 Final Summary (one line)

Ingress Controller relies on kube-proxy’s Service rules to receive traffic (NodePort) and to forward traffic to backend Pods (ClusterIP).
They don’t communicate directly, but they work together through Kubernetes Services.


---

If you want, I can also explain:

👉 Ingress vs Gateway API
👉 Packet flow diagrams (NodePort → Pod)
👉 How Cilium (eBPF) replaces kube-proxy in the Ingress path

Just tell me!

