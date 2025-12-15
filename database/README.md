 
 
# Database

 

### Relational Databases (SQL)

Use fixed schema **tables, rows, columns**

- Supports [ACID](oracle/ACID.md) transactions.
- [Practices to avoid anomalies](acid/anomalies.md)
- [Data should be Normalized](normalization/Normalization.md)
> Strongly Consistent and Available , does not support Partition Tolerence as it works on a single node.


* [Oracle](database/oracle/)
* [PostgreSQL](database/postgres/)
* [MySQL](database/mysql)
* SQL Server

Use cases: finance, ERP, OLTP, strong consistency.

## NoSQL
Not Only SQL 


Data distributed across multiple nodes and often across regions.
- Offers distributed ,sharded, replica-set databases.
- Offers [BASE](acid/Base.md) instead of ACID
- [Configurable Consistency , Availability and Partition Tolerence](cassandra/cap.md)
-  Document Databases (NoSQL) : Store **semi-structured JSON/BSON/XML** documents.
-  high availability, horizontal scalability.
-  flexible schema, product catalogs, user profiles.
-  Inbuilt router to distribute reads to read replicase 

Examples:
### Replicated Single-Leader
* [Mongo DB](database/mongodb/)
  

### Distributed, Multi-Leader or Leaderless Architecture
* [Cassandra](database/cassandra/)
* ScyllaDB
* Couchbase
* Riak KV
* RavenDB
* AWS DynamoDB
* Azure Cosmos DB
 

  
### In-Memory Databases

Keep data primarily in RAM for ultra-low latency , single instance used for in memory cache and Distributed Locking.

* Redis
* Memcached
 



## Columnar Databases

Store data **by columns** instead of rows. Very fast for analytics and aggregations. Used in data warehousing, BI, OLAP systems.

### Cloud Data Warehouses (PaaS)
- [Amazon Redshift](database/redshift): A fully managed, petabyte-scale data warehouse service that uses columnar storage.
- Google BigQuery: A serverless cloud data warehouse that uses a proprietary columnar format (Capacitor) for ultra-fast scans.
- Snowflake: A multi-cloud data platform that uses columnar storage within its micro-partitions.
- Microsoft Azure Synapse Analytics (formerly Azure SQL Data Warehouse).
- Oracle Autonomous Data Warehouse Cloud. 
### Analytics & Real-Time Databases (FOSS & Proprietary) 
- [ClickHouse](database/ClickHouse/): An open-source, high-performance, true columnar DBMS designed for real-time analytics and event data.
- Apache Druid: Designed for low-latency ingestion and fast analytical queries over massive time-series datasets, using a columnar format.
- Vertica: A commercial analytics database developed from the C-Store academic project, built as a pure column store.
- DuckDB: An in-process, embeddable columnar SQL OLAP RDBMS, excellent for local analytics.
- MonetDB: A pioneering open-source columnar relational DBMS.
- SAP HANA: An in-memory database that uses a hybrid model supporting both row and columnar storage.
- MariaDB ColumnStore: A storage engine for MariaDB that enables columnar storage. 
### Columnar File Formats (Storage Layer)
These are file formats used in big data ecosystems (like Hadoop and Spark) to store data columnarly, often queried by engines like Impala or Presto. 
- Apache Parquet: An open-source, widely adopted columnar storage format.
- Apache ORC (Optimized Row Columnar).
- Apache Kudu: A storage engine that provides fast analytic capabilities on top of Hadoop

 


##  Graph Databases

Data represented as **nodes + edges + properties**, ideal for relationships.

Examples:

* [Neo4j](database/neo4j/)
* Amazon Neptune
* JanusGraph

Use cases: fraud detection, social networks, recommendation engines.

 

##  Time-Series Databases

Optimized for **time-indexed data**, high-ingest rates.

Examples:

* InfluxDB
* Prometheus
* TimescaleDB

Use cases: metrics, IoT, monitoring, finance tick data.

   
 


 
##  Ledger / Blockchain Databases

Immutable, cryptographically verifiable transaction logs.

Examples:

* Amazon QLDB
* Hyperledger Fabric

Use cases: audit, regulatory compliance, tamper-proof records.
 
