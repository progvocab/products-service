

### AZ Failure Handling for AWS Lambda

AWS Lambda is **natively multi-AZ within a Region**.
You do not need to configure anything.

AWS automatically:

* Replicates function code across all AZs in that Region
* Schedules executions on healthy AZs
* Fails over transparently when an AZ goes down

**Your responsibility:**

* Ensure downstream dependencies are also multi-AZ (DynamoDB, SQS, ALB, API Gateway)
* Avoid VPC misconfiguration (multiple subnets across AZs)

**Best practice:**
Use **at least two subnets in different AZs** when attaching Lambda to a VPC.



### Region Failure Handling for AWS Lambda

Lambda does not automatically fail over across Regions.
You must design **multi-Region architecture**.

**How to handle Region failure:**

#### Pattern 1: Active-Passive Multi-Region with Route 53

* Deploy Lambda in **two Regions**
* Store state in a multi-Region service

  * DynamoDB global tables
  * S3 cross-Region replication
* Expose API with

  * Route 53 failover routing
  * API Gateway in each Region

When Region A fails, Route 53 health checks redirect to Region B.

#### Pattern 2: Active-Active Multi-Region

* Both Regions serve traffic simultaneously
* Use global data plane

  * DynamoDB global tables
  * S3 replication

Pros: Lowest RTO/RPO
Cons: Conflict resolution needed

#### Pattern 3: EventBridge Global Endpoint

* Primary Region + failover Region
* EventBridge global endpoint performs automatic failover for events
* Works well for event-driven Lambda

Example:

* Lambda consumes events
* EventBridge global endpoint fails over to secondary Region if primary fails


### Edge Location Failure Handling for Lambda@Edge

Lambda@Edge is replicated to **multiple Edge locations** automatically.

AWS automatically:

* Deploys the function to hundreds of PoPs
* Reroutes requests to the nearest healthy edge location
* No customer action required

If one Edge location fails:

* CloudFront routes user traffic to the next closest healthy edge node
* Lambda@Edge continues from the new location

**Your responsibility:**

* Ensure Lambdas are stateless
* Avoid calling Region-specific endpoints unless they are multi-Region


### Combined Strategy for High Resilience

#### Architecture that survives AZ, Region, and Edge failures:

* Lambda (multi-AZ automatic)
* API Gateway in two Regions
* Route 53 failover or latency routing
* DynamoDB global tables (multi-Region)
* S3 cross-Region replication
* CloudFront + Lambda@Edge for global access
* EventBridge global endpoints for event workflows

---

### Simple Real-World Example

#### Use case

Payment processing microservice using Lambda must never go down.

#### Solution

* **East-1** and **West-2** both host the Lambda + API Gateway
* DynamoDB global tables replicate state
* Route 53 failover
* CloudFront distributes client API calls
* EventBridge global endpoint sends events to healthy Region
* If an Edge PoP fails, CloudFront shifts traffic automatically
* If an AZ fails, Lambda automatically shifts
* If a Region fails, Route 53 + EventBridge shift execution to another Region

This design survives:

* Single AZ outage
* Complete Region disruption
* Edge location failure

 

More :

*  full multi-Region Lambda architecture
* A step-by-step deployment guide
* CloudFormation/Terraform snippets
