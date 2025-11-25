
# **AWS Billing Services**

AWS provides a full suite of services for **cost tracking, cost optimization, forecasting, budgeting, and chargeback/showback** in enterprises.


### **1. AWS Billing & Cost Management — Overview**

This is the **central dashboard** for anything related to billing, invoices, cost analysis, budgets, savings recommendations, credits, and cost alerts.

It includes:

* Bills & Payments
* Cost Explorer
* Budgets
* Cost & Usage Report (CUR)
* Cost Categories
* Reservation and Savings Plans details
* Tax invoices
* Payment method
* Consolidated billing (for AWS Organizations)



### **2. Core AWS Billing Terms & Concepts**



| Term                          | Meaning                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------- |
| **Payer Account**             | The management account paying all bills (in AWS Organizations).                 |
| **Linked Account**            | Member account under the payer account. Useful for chargeback/showback.         |
| **Cost Allocation Tags**      | Key-value tags added to resources to allocate cost by environment/team/project. |
| **Cost Categories**           | Logical grouping of costs (e.g., “Production”, “Analytics”, “Marketing Apps”).  |
| **AWS Free Tier**             | 12-month and Always-Free usage limits with billing alerts.                      |
| **Billing Alerts**            | SNS alerts for actual & forecasted cost thresholds.                             |
| **RI (Reserved Instances)**   | Commit to 1–3 years for EC2, RDS etc.  Up to 75% savings.                       |
| **Savings Plans**             | Commit to hourly spend ($/hr). More flexible than RIs.                          |
| **Spot Instances**            | Get spare capacity at up to 90% discount.                                       |
| **CUR (Cost & Usage Report)** | Detailed cost CSV/Parquet file (>10,000 columns). Used for analytics + audits.  |
| **Blended Cost**              | Average cost across accounts with reservations applied.                         |
| **Unblended Cost**            | Actual cost incurred by each account. Used for chargeback.                      |
| **Amortized Cost**            | Spreads upfront RI/Savings Plan cost across time.                               |
| **Effective Cost**            | Cost after RI/Savings Plan discounts are applied.                               |
| **Chargeback**                | Billing internal teams for the resources they consumed.                         |
| **Showback**                  | Showing cost to teams without charging them.                                    |



### **3. AWS Billing Services (Full List)**

| Service                                  | Purpose                                                     |
| ---------------------------------------- | ----------------------------------------------------------- |
| **AWS Billing Dashboard**                | Summary of charges, credits, and forecast.                  |
| **AWS Cost Explorer**                    | Visual graphs, usage analysis, RI/SP recommendations.       |
| **AWS Cost & Usage Report (CUR)**        | Most detailed billing dataset stored in S3 (hourly/daily).  |
| **AWS Budgets**                          | Custom budget alerts for cost, usage, RI/SP, and forecasts. |
| **AWS Cost Anomaly Detection**           | ML-powered detection of unusual spikes.                     |
| **AWS Price List API**                   | Programmatically fetch AWS pricing.                         |
| **AWS Marketplace Billing**              | Billing for purchased marketplace products/subscriptions.   |
| **AWS Reserved Instance Reporting**      | Track RI purchases, utilization, expiration.                |
| **Savings Plans Coverage & Utilization** | Track SP usage effectiveness.                               |



### **4. AWS Cost Explorer — Features**

AWS Cost Explorer is the primary UI to visualize cost.

### **Key Features:**

* Cost and usage graphs (daily, monthly, hourly)
* RI + Savings Plan **recommendations**
* Identify cost spikes
* S3 analytics (storage class analysis)
* Forecasting (up to 12 months)

### **Filters include:**

* Region
* Service
* Usage type
* Tags
* Linked account
* Cost category



### **5. AWS Budgets — Types & Features**

You can create budgets for:

### **Types of Budgets**

1. **Cost Budget**
2. **Usage Budget**
3. **RI Utilization Budget**
4. **RI Coverage Budget**
5. **Savings Plans Utilization Budget**
6. **Savings Plans Coverage Budget**


Below is a **concise, H3-only, no emojis** explanation of how to **set up each AWS Budget** and **how each one differs**.

### Cost Budget

**What it tracks**
Total AWS cost (blended or unblended) for one or more services, accounts, or tags.

**When to use**
You want to know: **“Notify me if spending exceeds X dollars.”**

**How to set up**

1. Open **AWS Billing Console → Budgets → Create budget**.
2. Select **Cost Budget**.
3. Set:

   * Budget amount (example: 500 USD)
   * Time period (monthly/quarterly)
   * Scope (linked account, service, tag)
4. Add **alert threshold** (example: 80%).
5. Choose **SNS/Email** recipients.
6. Create.


### Usage Budget

**What it tracks**
Tracks **amount of usage** such as:

* EC2 hours
* Lambda invocations
* S3 storage GB

**When to use**
You want: **“Alert me when EC2 hours exceed 1000 hrs.”**

**How to set up**

1. Budgets → **Create budget** → **Usage Budget**.
2. Select metric (like EC2, Lambda, S3).
3. Set numeric usage threshold.
4. Configure alerts via SNS/email.
5. Create.

 
### RI Utilization Budget

**What it tracks**
Percentage of **Reserved Instance utilization**.
Formula: `Used Hours / Purchased Hours`.

**When to use**
You want: **“Alert me if RI utilization drops below 90%.”**
This detects over-purchasing or unused RIs.

**How to set up**

1. Budgets → Create → **RI Utilization Budget**.
2. Select:

   * RI type (EC2, RDS, Redshift)
   * Utilization target (example: 90%).
3. Set alert thresholds.
4. Attach email/SNS.
5. Create.

 
### RI Coverage Budget

**What it tracks**
How much of your running workloads are **covered by existing RIs**.
Formula: `Covered Hours / Total Hours`.

**When to use**
You want: **“Alert me if my EC2 coverage falls below 70%.”**
This detects under-purchasing.

**How to set up**

1. Budgets → Create → **RI Coverage Budget**.
2. Select service (EC2, RDS).
3. Set coverage % target.
4. Add alerts → notify via email/SNS.
5. Create.

 

### Savings Plans Utilization Budget

**What it tracks**
Percentage of **Savings Plans utilization**.
Formula: `Used Commitment / Purchased Commitment`.

**When to use**
You want: **“Alert me if I’m using less than 85% of my Savings Plan.”**
Indicates unused commitment.

**How to set up**

1. Budgets → Create → **Savings Plans Utilization Budget**.
2. Choose Savings Plan.
3. Set target utilization (example: 85%).
4. Add alert threshold + SNS/email.
5. Create.

 

### Savings Plans Coverage Budget

**What it tracks**
How much of your **eligible compute usage** is covered by Savings Plans.
Example: How much EC2 usage is benefiting from SP discounts.

**When to use**
You want: **“Alert me if EC2 coverage drops below 60%.”**

**How to set up**

1. Budgets → Create → **Savings Plans Coverage Budget**.
2. Select service (EC2, Fargate, Lambda).
3. Set the desired coverage %.
4. Add alerts via SNS/email.
5. Create.

 

| Budget Type    | Tracks                           | Helps Detect         |
| -------------- | -------------------------------- | -------------------- |
| Cost Budget    | Total cost                       | Overspending         |
| Usage Budget   | Usage metrics                    | Overuse or spikes    |
| RI Utilization | % of RI used                     | Unused RIs           |
| RI Coverage    | How much usage is covered by RIs | Under-purchasing RIs |
| SP Utilization | % of SP commitment used          | Unused Savings Plans |
| SP Coverage    | How much usage is covered by SP  | Under-purchasing SP  |

 




### **Alerts (SNS / Email)**

* Actual cost
* Forecasted cost
* Thresholds (e.g., 80%, 100%)

### **Budget Actions**

You can automatically:

* Stop EC2
* Stop RDS
* Stop SageMaker
* Disable IAM keys
* Reduce DynamoDB throughput

(Used to protect against runaway cost.)



### **6. AWS Cost & Usage Report (CUR)**

The **CUR** is the **most accurate**, **fine-grained**, and **audit-level** cost data source.

### **Key Characteristics**

* Delivered to S3 bucket (CSV or Parquet)
* Contains 10,000+ columns
* Daily/hourly granularity
* Includes:

  * Resource ID
  * Tags
  * Usage quantity
  * Discounts (RI, SP)
  * Public/on-demand pricing
  * Effective cost
  * Credits & refunds

### **Common Integrations**

* **Athena** (query CUR with SQL)
* **Glue** (ETL)
* **Redshift** (analytics warehouse)
* **QuickSight** (BI dashboards)
* **EMR / Spark** (big data cost analysis)



### **7. AWS Cost Anomaly Detection**

Uses **machine learning** to detect unexpected cost spikes.

### **Features**

* Daily analysis
* Service-level anomaly detection
* SNS alerts
* Cost monitors (AWS service, Tag, Account-level)

Useful for avoiding surprise bills.



### **8. AWS Pricing Models**

| Model                  | Description               | Best for                         |
| ---------------------- | ------------------------- | -------------------------------- |
| **On-Demand**          | Pay per second/hour       | Unpredictable workload           |
| **Reserved Instances** | 1 or 3-year commitment    | Stable workloads                 |
| **Savings Plans**      | Commit to spend ($/hr)    | More flexible alternative to RIs |
| **Spot Instances**     | Up to 90% discount        | Fault-tolerant workloads         |
| **Dedicated Hosts**    | Physical server allocated | Compliance & licensing workloads |



### **9. AWS Tags for Cost Allocation**

AWS supports:

## **1. AWS-Generated Tags**

`aws:createdBy`

## **2. User-Defined Tags**

Example:

* `Environment=Prod`
* `Team=Analytics`
* `Project=CRM`

You must **activate** them in Billing Console to appear in reports.



### **10. Billing Alarms (CloudWatch + SNS)**

### **Steps**

1. Enable Billing Alerts
2. Create alarm on “EstimatedCharges”
3. Choose USD threshold
4. Add SNS → Email notifications

Note: **Available only in us-east-1**.



### **11. AWS Organizations — Cost Management**

### **Features**

* Consolidated billing
* Free tier pooled across accounts
* RI/SP sharing across accounts
* Cost allocation per account
* Combined discounts for volume pricing



#  **12. AWS Best Practices for Billing & Cost Optimization**





## **A. Tag All Resources**

* Use mandatory cost allocation tags
* Automate tagging (AWS Lambda / SCP)
* Enforce with AWS Organizations Service Control Policies (SCPs)



## **B. Use AWS Budgets and Alerts**

* Cost budget for overall cost control
* Usage budget (EC2 hours, S3 storage)
* Forecasted cost alerts

---

## **C. Identify Cost Spikes**

Enable **AWS Cost Anomaly Detection**

---

## **D. Use Reserved Instances or Savings Plans**

* For always-on servers
* Evaluate after 2 months of steady usage
* Prefer **Compute Savings Plans** (most flexible)

---

## **E. Clean Up Unused Resources**

* Stop unattached EBS volumes
* Delete unused snapshots
* Remove idle Load Balancers
* Clean old EIPs
* Shift S3 → IA / Glacier classes
* Optimize DynamoDB provisioned capacity

---

## **F. Use Spot Instances for Batch/ML workloads**

* EMR
* EKS with Spot Node Groups
* Auto Scaling groups with mixed instance types

---

## **G. Use S3 Lifecycle Policies**

Move data through:

* Standard → IA → Glacier → Deep Archive

Automatically reduce storage cost.

---

## **H. Use Compute Optimizer**

Provides ML-based:

* Instance right-sizing recommendations
* Lambda memory recommendations
* EBS volume recommendations

---

## **I. Use Cost Explorer for Daily Monitoring**

* Break down cost by service/team/project
* Identify anomalies early

---

## **J. Use CUR + Athena for Advanced Analytics**

Example:
"Show total EC2 cost for Production per month for 12 months."

---

## **K. Use Consolidated Billing**

* Share RI/SP discounts
* Centralized payer account
* Create separate accounts per team (best practice)



* **Cost Explorer** → Analyze and visualize cost
* **Budgets** → Alerts + automated actions
* **CUR** → Most detailed billing report
* **Cost Anomaly Detection** → ML-based alerts
* **Savings Plans / RIs** → Major cost reduction
* **Cost Categories & Allocation Tags** → Chargeback
* **Organizations** → Consolidated billing + RI/SP sharing



More :

 - A **diagram** explaining AWS Billing architecture
 - A **cheat sheet** for interview questions
- A **real-world cost optimization strategy**
- A **step-by-step guide to setting up billing alerts**


