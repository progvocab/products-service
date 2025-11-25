Here is the **concise, accurate list of AWS services that are *truly global* (not regional)**.
These services have a **single global control plane** and are **not bound to a specific AWS Region**.

### AWS Services That Are Global (Not Regional)

### 1. IAM (Identity and Access Management)

* Global users, roles, policies.
* No regional boundary.

### 2. Amazon Route 53

* Global DNS service.
* Hosted zones and records are global.

### 3. Amazon CloudFront

* Global CDN.
* Distributions are global objects.

### 4. AWS WAF (Web ACLs)

* Global when used with CloudFront.
* Note: Regional WAF also exists for ALB/API Gateway.

### 5. AWS Shield Standard & Advanced

* Global DDoS protection.

### 6. AWS Global Accelerator

* Global static anycast IPs routing to regional endpoints.

### 7. AWS Organizations

* Global governance across all accounts.

### 8. AWS Control Tower

* Global multi-account governance (though deployed in a home region).

### 9. AWS Artifact

* Global compliance reports.

### 10. AWS Support Center

* Global console for tickets and support plans.

### 11. AWS Marketplace

* Global software catalog.

### 12. ACM for CloudFront

* Certificates used with CloudFront are global.
* Note: ACM certificates in regions are regional, but **ACM for CloudFront is global**.

### 13. AWS IQ

* Global expert marketplace.

### 14. AWS Account Management / Billing / Cost Explorer

* Global (not tied to any region).

---

If you need the **list of regional vs. edge vs. hybrid** services too, ask and I’ll generate it.
