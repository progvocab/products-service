Below is a **concise, complete list of all AWS-recognized migration strategies (The 7 Rs)** with clear explanations.

### Rehost (Lift and Shift)

**What it is**
Move servers or VMs from on-prem to AWS **without architectural changes**.

**When to use**
Tight timelines, minimal changes, legacy apps.

**AWS Services**
AWS Application Migration Service (MGN), VM Import/Export.

---

### Replatform (Lift, Tinker and Shift)

**What it is**
Make **small optimizations** while migrating (database engine upgrades, OS update, switch to managed services).

**When to use**
You want better performance or lower ops overhead without rewriting the app.

**AWS Services**
RDS, Elastic Beanstalk, ElastiCache, Amazon MQ.

---

### Repurchase (Drop and Shop)

**What it is**
Move from an on-prem product to a **SaaS platform**.

**When to use**
Licensing is expensive, migration to managed SaaS is easier.

**Examples**
Jira Cloud, Salesforce, QuickBooks Online, Workday.

---

### Refactor / Re-architect

**What it is**
Redesign application to use **cloud-native architecture** (microservices, serverless, event-driven).

**When to use**
Scalability, agility, long-term modernization needed.

**AWS Services**
Lambda, API Gateway, DynamoDB, SQS, SNS, EventBridge, EKS.

---

### Relocate

**What it is**
Move **entire on-prem VMware clusters** to AWS **without changing the VMs**.

**When to use**
You want AWS benefits but must preserve VMware tools and management.

**AWS Services**
VMware Cloud on AWS.

---

### Retain

**What it is**
Keep certain applications on-prem temporarily or permanently.

**When to use**
Apps not ready for migration, regulatory restrictions, low ROI to move.

---

### Retire

**What it is**
Decommission apps that are no longer required.

**When to use**
Legacy systems with no business value.
 

| Strategy   | Description          | When to Use                   |
| ---------- | -------------------- | ----------------------------- |
| Rehost     | Move as-is           | Fastest migration             |
| Replatform | Minor optimizations  | Some benefits without rewrite |
| Repurchase | Switch to SaaS       | Replace legacy apps           |
| Refactor   | Rewrite for cloud    | Scalability + modernization   |
| Relocate   | Move VMware clusters | Keep VMware environment       |
| Retain     | Keep on-prem         | Not ready or low value        |
| Retire     | Decommission         | Cleanup unwanted systems      |

If you want, I can also provide:

* A migration strategy decision tree
* Example architecture diagrams
* Mapping of migration strategy by workload type (DB, app, batch, mainframe)
