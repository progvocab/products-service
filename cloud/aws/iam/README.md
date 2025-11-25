### What Is AWS IAM

AWS Identity and Access Management (IAM) is the global AWS service that controls **who** can access **what** in your AWS account and **how** they can access it. It defines identities (users, roles), permissions (policies), and authentication/authorization flows for all AWS services.

---

### IAM Core Terms and Concepts

### IAM User

A human identity with long-term credentials (password, access keys).
Used rarely in modern best practices (prefer roles instead).

### IAM Group

A collection of users for assigning common permissions.
Groups cannot have inline policies.

### IAM Role

A temporary identity assumed by:

* EC2 instances
* Lambda functions
* Containers
* Users (via SSO)
* External accounts
  Provides short-lived credentials via STS.

### Policy

A JSON document defining **allow/deny** rules.
Attached to a user, group, or role.

Example:

```json
{
  "Effect": "Allow",
  "Action": "s3:GetObject",
  "Resource": "arn:aws:s3:::my-bucket/*"
}
```

### Managed Policy

Predefined, reusable policies.
Two types:

* AWS-managed
* Customer-managed

### Inline Policy

Policy embedded directly inside a specific user/role.
Tightly coupled, cannot be reused.

### Trust Policy

Defines **who is allowed to assume a role**.
Attached to roles only.

Example:

```json
{
  "Effect": "Allow",
  "Principal": { "Service": "ec2.amazonaws.com" },
  "Action": "sts:AssumeRole"
}
```

### Permission Boundary

A guardrail that limits maximum permissions a role/user can ever have.
Used in multi-team environments.

### Service Control Policy (SCP)

Organization-wide guardrails applied at account or OU level.
Cannot grant access—only restrict.

### Session Policy

Additional inline permission applied to temporary credentials (STS).

### Identity Provider (IdP)

External SSO provider such as Azure AD, Okta, Google Workspace.
Used with IAM Identity Center / STS.

### STS (Security Token Service)

Issues short-lived credentials when roles are assumed.
 

### Additional Advanced IAM Concepts

### Resource-Based Policy

Attached directly to a resource (S3 bucket, SQS queue, Lambda).
Supports cross-account access.

### Condition Keys

Fine-grained access conditions (IP, VPC, MFA, encryption, tags).
Example:

```json
"Condition": { "Bool": { "aws:MultiFactorAuthPresent": true } }
```

### IAM Permissions Model

Authorization =
**Identity Policy** + **Resource Policy** + **Permissions Boundary** + **SCP**
All must allow the action (deny anywhere overrides allow).

---

### Best Practices for Securing AWS Using IAM

### Enforce Least Privilege

Give only the minimum permissions needed.
Prefer fine-grained actions instead of wildcards.

### Use IAM Roles, Not IAM Users

Replace users with IAM Identity Center or SSO.
Eliminate long-term access keys.

### Enable MFA Everywhere

MFA for console and API (via conditions).
Critical accounts: root, admin roles.

### Block Root User Use

Root should only be used for billing/initial setup.
Use CloudTrail alarms for root login.

### Use Permission Boundaries for Teams

Prevent developers or CI/CD pipelines from escalating privileges.

### Use SCPs in AWS Organizations

Enforce global rules like:

* No public S3
* No regions outside approved list
* No IAM:* permissions

### Restrict Access with Tags

Use ABAC (Attribute-Based Access Control) for scalable permission models.

### Enforce VPC-Only Access

Use IAM conditions:

* `aws:SourceVpce`
* `aws:SourceIp`
* `aws:SourceVpc`

Locks down S3, KMS, DynamoDB to private networks.

### Rotate Access Keys (If Any Exist)

Automate using IAM credential report or Config rules.

### Use IAM Access Analyzer

Detect unintended public or cross-account access.

### Log Everything

* CloudTrail for all API calls
* CloudWatch alarms for suspicious API events
* S3 server access logs for sensitive buckets

### Never Use Inline Policies for Admin Access

Use customer-managed policies for traceability and versioning.

### Encrypt Everything by Default

Use KMS keys with key policies restricting access.

 

More:

* A **diagram showing how IAM evaluates permissions**
* A **table comparing policy types**
* **Real-world IAM security architecture** for enterprise AWS accounts


### **Setting Up a Permission Boundary**

 

A **Permission Boundary** is a *maximum permission limit* attached to IAM users or roles.
Even if their attached policies allow more, they **cannot exceed** the boundary.
Used in multi-team environments and CI/CD pipelines to prevent privilege escalation.



### How to Create a Permission Boundary

### Step 1: Create a Customer-Managed Policy

Go to **IAM → Policies → Create Policy**.

Example boundary allowing only S3 read and Lambda invoke:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "lambda:InvokeFunction"
      ],
      "Resource": "*"
    }
  ]
}
```

Save as:
`MyTeamPermissionBoundary`


### Step 2: Apply Boundary to a Role or User

Go to: **IAM → Roles → Create Role** (or edit an existing role).
Under **Permissions boundary**, choose:

**Use a permissions boundary to control...**
Select **MyTeamPermissionBoundary**.

Now that role can never exceed the allowed actions, even if another policy grants more.


### Step 3: Verify Boundaries Are Working

Effective permissions =
**Intersection of** (identity policies ∩ permission boundary).

If boundary denies an action, user/role cannot do it.


### Best Uses

* Developers allowed to create IAM roles but not escalate privileges
* CI/CD automation roles
* Multi-team governance
* SaaS tenants inside same AWS account


### **Setting Up a Service Control Policy (SCP)**

### What It Is

A **Service Control Policy** is an AWS Organizations feature that sets *account-level guardrails*.
SCPs apply to:

* Entire Organization
* Organizational Units (OUs)
* Individual Accounts

SCPs do NOT grant permissions; they only restrict.


### Step-by-Step: Creating an SCP

### Step 1: Enable AWS Organizations

Go to **AWS Organizations** → create an organization (if not already).

### Step 2: Create an OU (Optional but recommended)

Example OUs:

* Prod
* Dev
* Restricted

### Step 3: Create the SCP

Go to: **AWS Organizations → Policies → Service Control Policies → Create Policy**

Example SCP: **Deny IAM Changes**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": "iam:*",
      "Resource": "*"
    }
  ]
}
```

Example SCP: **Block Non-Approved Regions**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:RequestedRegion": [
            "us-east-1",
            "eu-west-1"
          ]
        }
      }
    }
  ]
}
```

### Step 4: Attach SCP to an OU or Account

Go to:
**Organizations → OUs → Select OU → Attach SCP**

SCP is now enforced for all IAM roles, users, services inside that account.


### Step 5: Validate

Use **IAM Access Analyzer** or test in the account: denied actions appear as:

`explicit deny from organizations SCP`

 

### **Key Differences Between Permission Boundary and SCP**

| Concept                 | Applies To              | Purpose                                 | Enforced By       |
| ----------------------- | ----------------------- | --------------------------------------- | ----------------- |
| **Permission Boundary** | Specific IAM user/role  | Limit max permissions                   | IAM               |
| **SCP**                 | Whole AWS account or OU | Restrict what the account itself can do | AWS Organizations |
| Must allow?             | Yes, defines max allow  | No, only denies                         |                   |
| Who uses it?            | Multi-team IAM safety   | Enterprise governance                   |                   |
 
### **Best Practices**

### For Permission Boundaries

* Use boundaries for dev teams who can create IAM roles
* Use boundaries for CI/CD roles
* Store boundaries as customer-managed policies
* Never allow wildcard admin actions in boundaries

### For SCPs

* Start with AWS-provided **"Guardrails"** (best practice SCP sets)
* Test SCPs in a sandbox OU first
* Use SCPs to block dangerous global APIs:

  * `iam:*`
  * `kms:ScheduleKeyDeletion`
  * `ec2:DeleteVpc`
  * Non-approved Regions
  * Public S3 access
* Always allow AWS logging services:

  * CloudTrail
  * Config
  * Security Hub
  * GuardDuty
  * S3 logging
 

More:

* A **Mermaid diagram** showing how IAM policies + boundaries + SCPs evaluate together
* A table of **common enterprise SCPs** used in production
* A sample **multi-account architecture** with OUs, boundaries, and SCPs


### **What Is a Trust Policy**

A **Trust Policy** is a JSON document attached to an IAM **Role** that defines **who is allowed to assume the role**.
It is evaluated by **AWS STS** during `sts:AssumeRole` or service role assumption.

It answers the question:
**“Who can use this role?”**
 

### **How to Set Up a Trust Policy**

### Step 1: Create an IAM Role

Go to **IAM → Roles → Create role**.

Choose:

* **AWS service** (e.g., EC2, Lambda)
  or
* **Another AWS account**
  or
* **Web identity / SAML**

AWS will generate a trust policy automatically, but you can edit it.
 

### **Step 2: Edit the Trust Policy (JSON)**

Under **Trust relationships → Edit trust policy**, define the principal allowed to assume the role.

 

### **Common Trust Policy Types**

### 1. **Service Role (EC2, Lambda, ECS, etc.)**

Allows AWS service to assume the role.

Example: EC2 Instance Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Used by: **EC2 Instance Metadata Service (IMDS)** to fetch temporary credentials.

 

### 2. **Cross-Account Role**

Allows another AWS account to assume the role.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:root"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Use case:
A central governance account assumes roles in spoke accounts.

 
### 3. **Restrict by External ID (Best Practice for Third-Party Access)**

Prevents confused-deputy attacks.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "MY_SECURE_ID_12345"
        }
      }
    }
  ]
}
```

Common for vendors (Datadog, Snowflake, etc.).

 

### 4. **Web Identity (OIDC / Cognito / EKS IAM Roles for Service Accounts)**

Used in EKS IRSA.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/ABCDE12345"
      },
      "Action": "sts:AssumeRoleWithWebIdentity"
    }
  ]
}
```

Used by:
Pods via **kubelet** calling `sts:AssumeRoleWithWebIdentity`
 

### 5. **SAML (Enterprise SSO)**

For corporate Active Directory or Okta.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:saml-provider/MyIDP"
      },
      "Action": "sts:AssumeRoleWithSAML"
    }
  ]
}
```

Used by:
Employees logging into AWS Console via SSO.

 

### **Step 3: Apply Least-Privilege Conditions (Recommended)**

You can add constraints such as:

#### Restrict by source VPC endpoint

```json
"Condition": { "StringEquals": { "aws:SourceVpce": "vpce-12345" } }
```

#### Force MFA (for human users)

```json
"Condition": { "Bool": { "aws:MultiFactorAuthPresent": "true" } }
```

 
### **Step 4: Test the Role**

Use:

```bash
aws sts assume-role --role-arn arn:aws:iam::111111111111:role/ExampleRole --role-session-name test
```

If trust policy is wrong → error: **“Access denied”**.

 

### **Summary Table**

| Use Case          | Principal Type                   | Example Action                  |
| ----------------- | -------------------------------- | ------------------------------- |
| EC2/Lambda/ECS    | `Service:`                       | `sts:AssumeRole`                |
| Cross-account     | `AWS:`                           | `sts:AssumeRole`                |
| Vendor access     | `AWS:` + `Condition: ExternalId` | `sts:AssumeRole`                |
| Kubernetes / OIDC | `Federated:`                     | `sts:AssumeRoleWithWebIdentity` |
| SSO / SAML        | `Federated:`                     | `sts:AssumeRoleWithSAML`        |

 

More :

* A **Mermaid diagram showing trust policy flow**
* A **full cross-account IAM setup** with permissions, SCP, and trust policy
* A **real-world IAM role architecture** for enterprise workloads
