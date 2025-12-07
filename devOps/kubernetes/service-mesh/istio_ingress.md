Yes — Istio includes its own Ingress implementation called the Istio Ingress Gateway.
But it is not the same as the standard Kubernetes Ingress.

Let’s break this down clearly.


---

✅ Istio has a built-in Ingress called the Istio Ingress Gateway

When you install Istio (default profile), it deploys:

istio-ingressgateway (Envoy proxy)

istiod (control plane)


The Ingress Gateway is:

an Envoy proxy deployed as a Kubernetes Service

usually exposed via a LoadBalancer service

responsible for all incoming traffic into the mesh


So yes: Istio includes its own ingress component.


---

🆚 Istio Ingress Gateway vs Kubernetes Ingress

1. Kubernetes Ingress

Uses the Kubernetes Ingress API

Needs an Ingress controller like NGINX, AWS ALB, Traefik

Provides basic L7 routing

Limited to HTTP/HTTPS

No mTLS between services

No mesh-level routing capabilities


2. Istio Ingress Gateway

Uses Istio Gateway + VirtualService CRDs (not Kubernetes Ingress API)

Built on top of Envoy proxy

Supports HTTP, HTTP/2, gRPC, TCP, TLS, mTLS

Full traffic shaping:

canary

blue/green

traffic splitting

header-based routing

retries, timeouts, circuit-breaking

mutual TLS at the edge


Integrated with the service mesh



---

🟦 Istio Has Its Own API Instead of Kubernetes Ingress API

To configure the Istio Ingress Gateway, you use:

1. Gateway

Defines ports, protocols, and TLS termination.

Example:

apiVersion: networking.istio.io/v1alpha3
kind: Gateway

2. VirtualService

Defines traffic routing rules.

Example:

apiVersion: networking.istio.io/v1alpha3
kind: VirtualService

Unlike K8s Ingress, Istio Ingress is much more powerful and deeply integrated with Envoy.


---

🟩 Mermaid: How Traffic Comes Into Istio

graph TD

External[External Client] --> LB[AWS/NLB/ELB LoadBalancer]
LB --> IGW[Istio Ingress Gateway (Envoy)]

IGW -->|Routing via VirtualService| App1[Service A Pod+Sidecar]
IGW --> App2[Service B Pod+Sidecar]

App1 --> App2


---

⭐ Key Points to Remember

✔️ Istio does NOT use Kubernetes Ingress (by default)

Istio uses its own CRDs for routing.

✔️ Istio ships with its own Ingress Gateway (Envoy)

This replaces NGINX or AWS ALB Ingress.

✔️ You can make Istio work with Kubernetes Ingress

But that is optional and rare.

✔️ Using Istio’s Gateway gives far more control

Better security (mTLS), routing, observability, and reliability features.


---

🟧 Summary

Feature	Kubernetes Ingress	Istio Ingress Gateway

Implemented by	NGINX/ALB/etc	Envoy
Config API	Ingress	Gateway + VirtualService
Routing features	Basic	Advanced (L7, retries, canary, etc.)
Works with mesh	❌	✔️
Requires Envoy	❌	✔️



---

If you want, I can also explain:

✅ How Istio Ingress Gateway interacts with CNI
✅ How traffic flows inside the mesh after the gateway
✅ Difference between Ingress Gateway and Service Mesh Gateways (East-West Gateway)

To map a domain name (e.g., api.example.com) to an Istio Ingress Gateway host, you must configure DNS → LoadBalancer → Istio Gateway → VirtualService.

Here is the correct and complete workflow.


---

✅ Step 1: Get the Istio Ingress Gateway External IP

Istio installs an Ingress Gateway as a Service:

kubectl get svc -n istio-system istio-ingressgateway

You'll see something like:

NAME                   TYPE           EXTERNAL-IP      PORT(S)
istio-ingressgateway   LoadBalancer   34.201.25.92     80:31380/TCP

✔️ The EXTERNAL-IP (34.201.25.92) is the public entrypoint.

> On AWS EKS, the gateway is usually backed by an NLB or ELB.




---

✅ Step 2: Point Your Domain DNS to the Gateway External IP

Go to your DNS provider (Route53, GoDaddy, Cloudflare, etc.) and create:

A Record

Host: api.example.com
Type: A
Value: 34.201.25.92     ← Istio ingress external IP
TTL: 300

If using Cloudflare, make sure the DNS proxy (orange mode) is either off or compatible.

Now api.example.com → (DNS) → Istio Gateway IP.


---

✅ Step 3: Configure an Istio Gateway

You must configure a Gateway resource to accept HTTP/S traffic for that domain.

Example:

apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: my-gateway
  namespace: default
spec:
  selector:
    istio: ingressgateway  # use istio default ingress gateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "api.example.com"

✔️ The hosts field must exactly match your domain name
✔️ The selector must match the label of Istio’s ingress gateway


---

✅ Step 4: Route the Domain to Your Backend Service

Use a VirtualService:

apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: my-service-routing
  namespace: default
spec:
  hosts:
    - "api.example.com"
  gateways:
    - my-gateway
  http:
    - route:
        - destination:
            host: my-service
            port:
              number: 80

Now the flow is:

api.example.com → (DNS) → IngressExternalIP → Istio Gateway → VirtualService → Kubernetes Service (my-service) → Pod


---

📌 Important Notes

✔️ DNS A record must point to the LoadBalancer IP of Istio Ingress

If you are on AWS and using NLB, you can use:

A record → NLB hostname
OR

CNAME → NLB hostname (recommended)


Example:

api.example.com CNAME a1b2c3d4e5f6.elb.amazonaws.com


---

✔️ Optional: HTTPS with TLS

Istio supports certificates via:

Manual TLS secret

Let’s Encrypt (via cert-manager)

Custom CA


If you want, I can show you:

✔️ Istio Gateway with TLS
✔️ Automatic Let’s Encrypt using cert-manager
✔️ How to handle multiple domains/subdomains


---

🎯 Summary

To map a domain to Istio Ingress:

1. Get Istio ingress external IP


2. Create DNS record → point domain to that IP


3. Create Istio Gateway with hosts: "your-domain"


4. Create VirtualService routing rules


5. Optional: add TLS certificates




---

If you want, I can generate the full production-grade setup (HTTPS + autoscaling + health checks + retries).