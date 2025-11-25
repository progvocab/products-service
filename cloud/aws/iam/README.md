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

---

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

---

More:

* A **diagram showing how IAM evaluates permissions**
* A **table comparing policy types**
* **Real-world IAM security architecture** for enterprise AWS accounts
