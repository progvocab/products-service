# Disaster Recovery

### **Use Multi-AZ as the Baseline**

Always deploy compute, databases, load balancers, and storage across **multiple Availability Zones** inside a Region.
This protects from AZ outages and forms the foundation of all DR.



### **Choose the DR Strategy Based on RTO/RPO**

Select Backup & Restore, Pilot Light, Warm Standby, Active–Passive, or Active–Active based on business-defined **Recovery Time Objective** and **Recovery Point Objective**.
Do not over-engineer; match DR to the application’s criticality.



### **Automate Infrastructure Using IaC**

Use **CloudFormation**, **CDK**, or **Terraform** to recreate the entire stack in a DR Region.
Manual steps should be zero.


### **Automate Cross-Region Replication**

Enable continuous replication for critical data:

* S3 CRR or multi-Region buckets
* DynamoDB Global Tables
* Aurora Global Database
* ECR image replication
* KMS multi-Region keys

Ensures you meet RPO requirements.


### **Use Global DNS for Failover**

Use **Amazon Route 53**:

* Health checks
* Failover routing
* Weighted routing
* Latency-based routing

Enables automatic redirection to DR Region during failure.


### **Design Stateless Application Tiers**

Ensure app servers (EC2, ECS, Lambda functions) do not depend on local state.
State goes in S3, DynamoDB, Redis Global Datastore, or Aurora Global.

This makes redeployment in DR Region trivial.



### **Version and Replicate Application Artifacts**

Replicate:

* Lambda versions
* Container images (ECR)
* AMIs
* Configuration via SSM Parameter Store

Artifacts must be available in the DR Region to bootstrap services instantly.


### **Use Event-Driven Asynchronous Workflows**

Queue-based architecture improves durability and DR performance:

* SQS with DLQ
* SNS
* EventBridge (with archive)
* Kinesis with multi-Region replication

Allows the system to continue operating even when downstream services fail.

 

### **Leverage Managed Serverless Services**

Prefer services that provide built-in DR across AZs:

* Lambda
* DynamoDB
* API Gateway
* S3
* Aurora Serverless v2

Reduces the operational burden of DR.

 

### **Practice DR With Game Days**

Run periodic failover drills:

* AZ failure simulation
* Region failover simulation
* DNS failover rehearsal
* Data consistency validation
* Cold restore time measurement

Ensures your DR plan actually works.

 

### **Monitor Replication and Failover Readiness**

Track:

* S3 CRR lag
* DynamoDB replication health
* Aurora replica lag
* Route 53 health checks
* Kinesis replication status
* CloudWatch Alarms on critical paths

Ensures you don't discover issues during an outage.

 

### **Secure Your DR Setup**

* Use cross-account backup
* Encrypt data using multi-Region KMS keys
* Restrict IAM access
* Ensure DR Region uses the same guardrails (Config, CloudTrail, IAM boundaries)

DR must not introduce security gaps.

 

### **Cost Optimization**

* Use lifecycle policies for backups
* Tier data to Glacier
* Choose Warm Standby or Pilot Light instead of Active-Active when possible
* Turn off non-critical DR components out of business hours

Ensures DR is affordable.

 

### **Summary of Key Best Practices**

1. Always Multi-AZ first
2. Pick DR strategy based on RTO/RPO
3. Automate infra + failover
4. Replicate all critical data
5. Make apps stateless
6. Use global DNS failover
7. Practice DR regularly
8. Monitor replication health
9. Secure DR as strongly as primary
10. Optimize cost without affecting RTO/RPO


### **Backup and Restore**

A DR strategy where you frequently back up data (Amazon S3, EBS snapshots, RDS snapshots) and restore it into a new environment after a disaster.
**RTO:** Hours
**RPO:** Minutes–Hours
**Use case:** Low-cost DR for non-critical systems.
**Example:** Nightly RDS snapshot → restore in another Region when outage occurs.


### **Pilot Light**

A minimal version of your environment is always running in the DR Region. Only core components (databases, IAM, small EC2 instances, minimal infrastructure) are active. Application servers scale up only during a disaster.
**RTO:** Minutes
**RPO:** Minutes
**Use case:** Databases must be warm, app servers can start on demand.

**Example Flow (AWS):**

* DynamoDB global tables → always on
* Minimal EC2/AMI or Lambda deployed
* During disaster → Auto Scaling launches full fleet
* Route 53 failover shifts traffic

### **Warm Standby**

A scaled-down but fully functional version of your application is running in another Region.
Compared to Pilot Light, more components run continuously (but smaller capacity).
**RTO:** Seconds–Minutes
**RPO:** Seconds
**Use case:** Medium to high-importance workloads.

**Example:**

* 2-node RDS read replica in DR Region
* Service deployed at 30% capacity
* During disaster → scale to 100%


### **Active–Passive (Multi-Region Failover)**

One Region handles all traffic; the second Region is fully provisioned but idle.
Failover happens using Route 53 health checks.
**RTO:** Seconds
**RPO:** Near-zero (if DB replicated).
**Use case:** High availability without high cost of active-active.

**Example:**

* Aurora Global Database: primary in us-east-1, secondary in eu-west-1
* Route 53 failover → send traffic to secondary Region


### **Active–Active Multi-Region**

Both Regions serve traffic simultaneously.
Requires globally consistent data replication (e.g., DynamoDB Global Tables, S3 Cross-Region Replication + Iceberg metadata sync).
**RTO:** Zero
**RPO:** Zero or near-zero
**Use case:** Mission-critical workloads that cannot tolerate downtime.

**Example:**

* API Gateway + Lambda deployed in both Regions
* DynamoDB Global Tables
* Route 53 latency-based routing
* Failover occurs automatically


### **Multi-AZ Resilience (Not a DR strategy, but foundational)**

Handles failure of an Availability Zone inside the same Region.
Every production architecture must use Multi-AZ.
**RTO:** Zero
**RPO:** Zero

**AWS example:**

* RDS Multi-AZ
* Lambda is automatically Multi-AZ
* ALB is Multi-AZ

Not enough for Regional DR but solves AZ failures.


### **Cloud-Native Disaster Recovery Techniques**

### **1. Cross-Region Replication (CRR)**

Applies to S3 buckets, DynamoDB, Aurora, ECR, AMIs, etc.
Part of Pilot Light / Warm Standby / Active–Active.

### **2. Route 53 Failover**

Health checks + weighted/latency-based records to redirect traffic globally.

### **3. Infrastructure-as-Code Redeployment**

Rebuild environment using CloudFormation, CDK, Terraform.
Used heavily in Backup & Restore.


### **Summary Table**

| DR Strategy          | RTO             | RPO       | Cost                     | When to Use                                    |
| -------------------- | --------------- | --------- | ------------------------ | ---------------------------------------------- |
| **Backup & Restore** | Hours           | Hours     | Lowest                   | Non-critical workloads                         |
| **Pilot Light**      | Minutes         | Minutes   | Low-Medium               | Databases must stay warm                       |
| **Warm Standby**     | Seconds–Minutes | Seconds   | Medium                   | High-availability apps with budget constraints |
| **Active-Passive**   | Seconds         | Near-zero | Medium-High              | High availability with low traffic duplication |
| **Active-Active**    | Zero            | Zero      | Highest                  | Mission-critical global applications           |
| **Multi-AZ**         | Zero            | Zero      | Included in service cost | AZ-level resiliency                            |


