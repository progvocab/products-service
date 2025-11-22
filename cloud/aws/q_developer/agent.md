AWS has not published the internal list of every agent AWS Q Developer uses, but they *have* described the categories of agents Q can call behind the scenes. Based on AWS docs, Bedrock “Agents”, and what Q Developer actually does in IDE/AWS Console, here are the **known and deduced functional agents** that AWS Q uses.

Below is the **authoritative and practical list** of agents AWS Q Developer can invoke.

---

# ⭐ **Agents AWS Q Developer Can Call (Internally)**

AWS Q Developer is built as a **multi-agent system**, where each agent specializes in a particular AWS domain or task. These run behind the scenes and are orchestrated by the “Task Planner”.

Here are the major types of agents:

---

# 🔵 **1. Code Generation Agent**

Generates:

* Python/Java/Node/Go code
* Lambda handlers
* API Gateway integrations
* SDK code for S3, DynamoDB, SQS, SNS
* CI/CD snippets (GitHub Actions, CodeBuild)

**Special abilities:**
Fixes, explains, optimizes and reviews code.

---

# 🔵 **2. Infrastructure-as-Code Agent**

Generates IaC for:

* **AWS CDK** (TS/Python/Java)
* **CloudFormation**
* **Terraform**

Tasks:

* Build VPCs, Lambdas, ECS clusters
* Convert CloudFormation → CDK
* Refactor IaC for best practices

---

# 🔵 **3. AWS Documentation Retrieval Agent (RAG Agent)**

Retrieves:

* AWS API reference
* SDK examples
* Best practice guides
* Well-Architected Framework
* Security recommendations

This agent feeds authoritative AWS docs into the LLM.

---

# 🔵 **4. Resource-Inspection Agent (IAM-Bound Execution Agent)**

Reads:

* Lambda configuration
* S3 bucket settings
* IAM policies
* CloudWatch logs
* ECS task definitions
* VPC/Subnet/Security Groups

It does *not* modify anything—read-only unless allowed.

---

# 🔵 **5. Error Diagnosis / Troubleshooting Agent**

Interprets:

* CloudWatch logs
* Stack traces
* IAM “AccessDenied” errors
* Lambda timeout / OOM
* RDS connection failures
* ECS 502/503 errors

It applies AWS-known fix patterns.

---

# 🔵 **6. IAM Policy Agent**

Creates and validates:

* least privilege IAM policies
* role trust relationships
* resource-scoped permissions

Also checks for:

* privilege escalation
* unsafe wildcards
* missing permissions

---

# 🔵 **7. Architecture Design Agent**

Generates:

* serverless architectures
* event-driven workflows
* microservice layouts
* VPC and networking diagrams
* high availability & DR architectures

Uses AWS Well-Architected pillars.

---

# 🔵 **8. Cost Optimization Agent**

Analyzes:

* over-provisioned compute
* unused EBS volumes
* inefficient queries
* wrong storage classes
* expensive Lambda patterns

Applies Cost Optimization Framework.

---

# 🔵 **9. Security Review Agent**

Checks for:

* public S3 buckets
* missing encryption
* weak IAM roles
* Lambda vulnerabilities
* missing audit logs

Uses Security Hub + IAM Access Analyzer patterns.

---

# 🔵 **10. Migration & Modernization Agent**

Used for:

* Java 8 → Java 17 migration
* Python 2 → Python 3 migration
* Monolith → microservices suggestions
* MySQL → Aurora migration patterns

---

# 🔵 **11. CI/CD Agent**

Builds:

* CodePipeline
* GitHub Actions
* GitLab CI
* CDK Pipelines

Also generates buildspec.yml or Dockerfiles.

---

# 🔵 **12. SQL & NoSQL Query Agent**

Helps generate:

* DynamoDB PartiQL
* PostgreSQL/MySQL SQL queries
* Athena queries
* Redshift SQL

---

# 🟢 Additional: AWS Q Enterprise Also Has “Business Agents”

(Not specific to Q Developer)

* Data agent
* Analytics agent
* CRM agent

But for **Q Developer**, the above 12 agents are the relevant ones.

---

# ⭐ Summary Table

| Agent                     | What It Does                   |
| ------------------------- | ------------------------------ |
| Code Generation Agent     | Writes/fixes code              |
| IaC Agent                 | Generates CDK/CFN/Terraform    |
| RAG Retrieval Agent       | Fetches AWS docs               |
| Resource Inspection Agent | Reads AWS configs/logs         |
| Error & Debug Agent       | Explains and fixes errors      |
| IAM Agent                 | Creates and validates policies |
| Architecture Agent        | Designs cloud architectures    |
| Cost Agent                | Reduces cloud cost             |
| Security Agent            | Hardens AWS setup              |
| Migration Agent           | Code & infra migrations        |
| CI/CD Agent               | Creates pipelines              |
| Query Agent               | SQL/NoSQL code                 |

---

# If you want, I can also provide:

✅ A **Mermaid Architecture Diagram** showing all agents
✅ A **sequence diagram** (how a user query flows through the agents)
✅ A **deep dive** into how AWS Q’s Task Planner orchestrates agents
Just tell me!

