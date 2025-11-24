Here are **high-level, deep, difficult DynamoDB questions** that are realistic for **AWS Solutions Architect interviews**.
All are designed to test **architecture thinking, partition behavior, consistency, indexing, throttling, scaling, and cost optimization**.

### ### 1. Partition & Hot Key Analysis

Your DynamoDB table receives 2,000 writes/sec for a single employee record because a GPS system updates “EmployeeId = X” constantly. The table is in on-demand mode.
Explain:

* What happens at the partition level
* Which DynamoDB internal components are impacted
* Whether adaptive capacity can help
* How you would redesign the keys

### ### 2. Read Consistency vs Cost Optimization

A real-time dashboard reads the latest status of 30,000 employees every second.
The system uses strongly consistent reads and costs are skyrocketing.
Explain:

* Cost difference
* How to redesign the read pattern
* What consistency tradeoffs you would choose
* How leader replicas are impacted

### ### 3. Single Table Design Query Explosion

You designed a single table with this key:
`PK = EMP#<EmployeeId>`, `SK = <RecordType>#<Timestamp>`
A new use case requires querying all employees by DepartmentId with a timestamp filter.
Explain:

* Why your current model breaks
* How to fix it with GSIs
* How GSIs create their own partitions
* What the cost and throttling risks are

### ### 4. Impact of GSIs on Write Capacity

An Employees table has 4 GSIs. A batch job writes 10,000 employee updates/minute.
Explain:

* How write amplification works
* How Streams and GSIs multiply WCU consumption
* Under what scenario the GSI becomes the bottleneck instead of the base table

### ### 5. Adaptive Capacity Behavior

Your table has 20 partitions, but one partition receives 80% of traffic.
Explain:

* How adaptive capacity redistributes throughput
* When it fails to help
* Why adaptive capacity only helps hot partition keys, not hot items
* Why throughput spikes still cause throttling

### ### 6. On-Demand Scaling Limits

An e-commerce site spikes from 50 writes/sec to 5,000 writes/sec in under 1 second.
The table is configured for on-demand.
Explain:

* Why on-demand may still throttle
* How DynamoDB determines burst limits
* What happens inside the partition allocator

### ### 7. 10GB Partition Size Split Scenario

Your Employees table sees explosive growth in the “EmployeeProfile” item.
It passes 10GB and DynamoDB splits the partition.
Explain:

* How partitions are split
* How read/write throughput is redistributed
* Whether hot patterns remain hot after splitting
* How partition keys change behind the scenes

### ### 8. Complex Transactional Workloads

Your application uses `TransactWriteItems` to update 3 items per employee update.
Suddenly you experience throttling.
Explain:

* Why each transaction consumes 2× read & write capacity
* How DynamoDB handles transaction coordination
* What design change reduces cost and throttling

### ### 9. Multi-Region Global Table Conflict Resolution

You enable Global Tables for cross-region write capability.
Two regions simultaneously update the same employee record.
Explain:

* What happens internally
* How "last writer wins" timestamp resolution works
* How to design the application to avoid data loss

### ### 10. Large Item Access Patterns

A table stores Resumes (200 KB) for each employee.
A search API frequently scans the table.
Explain:

* Why scans become extremely expensive
* Why large item size hurts partition performance
* How to use S3 off-loading and store metadata only

---

If you want, I can:

* Provide **ideal answers** for each question
* Generate **even more difficult scenario-based questions**
* Create a **mock interview** with 20–30 DynamoDB architect-level questions
