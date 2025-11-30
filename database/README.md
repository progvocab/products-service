 
 
# Database

 

### Relational Databases (SQL)

Use fixed schema **tables, rows, columns**

- Supports [ACID](oracle/ACID.md) transactions.
- [Practices to avoid anomalies](acid/anomalies.md)
- [Data should be Normalized](normalization/Normalization.md)
- Strongly Consistent and Available , does not support Partition Tolerence as it works on a single node.
- 

* Oracle
* PostgreSQL
* MySQL
* SQL Server

Use cases: finance, ERP, OLTP, strong consistency.

## NoSQL

### Distributed Databases

Data distributed across multiple nodes and often across regions.
- Offers distributed ,sharded, replica-set databases.
- Offers [BASE](acid/Base.md) instead of ACID
- [Configurable Consistency , Availability and Partition Tolerence](cassandra/cap.md)

Examples:
* Mongo DB
* Cassandra
* Google Spanner
* Hadoop HBase

Use cases: high availability, horizontal scalability.

### Document Databases (NoSQL)

Store **semi-structured JSON/BSON/XML** documents.

Examples:

* MongoDB
* Couchbase
* Amazon DocumentDB

Use cases: flexible schema, product catalogs, user profiles.
 
### Key–Value Databases (NoSQL)

Store simple pairs: **key → value**. Extremely fast and horizontally scalable.
 
* Redis
* Amazon DynamoDB
* Riak

Use cases: caching, sessions, high-scale lookups.
  
### In-Memory Databases

Keep data primarily in RAM for ultra-low latency.

Examples:

* Redis
* Memcached
* SAP HANA

Use cases: caching, real-time apps.



### Columnar Databases

Store data **by columns** instead of rows. Very fast for analytics and aggregations.

Examples:

* Amazon Redshift
* Apache Cassandra
* Apache HBase
* Google BigQuery

Use cases: data warehousing, BI, OLAP systems.



### Graph Databases

Data represented as **nodes + edges + properties**, ideal for relationships.

Examples:

* Neo4j
* Amazon Neptune
* JanusGraph

Use cases: fraud detection, social networks, recommendation engines.

 

### Time-Series Databases

Optimized for **time-indexed data**, high-ingest rates.

Examples:

* InfluxDB
* Prometheus
* TimescaleDB

Use cases: metrics, IoT, monitoring, finance tick data.

   


---


 
### Ledger / Blockchain Databases

Immutable, cryptographically verifiable transaction logs.

Examples:

* Amazon QLDB
* Hyperledger Fabric

Use cases: audit, regulatory compliance, tamper-proof records.

 

Tell me which version you want.
