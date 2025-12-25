Partition pruning is a database optimization technique where the query optimizer skips irrelevant data partitions based on  clause filters, significantly reducing I/O and speeding up queries by only scanning necessary data files, crucial for large, partitioned datasets in data warehouses and big data systems like Oracle



 (https://docs.oracle.com/cd/E11882_01/server.112/e25523/part_avail.htm), Spark, BigQuery, and Impala (https://docs.cloudera.com/cdw-runtime/cloud/impala-reference/topics/impala-partition-pruning.html). It works by analyzing partition keys (like date, region) and efficiently filtering partitions using static (compile-time) or dynamic (run-time) logic, improving performance by avoiding full table scans. [1, 2, 3, 4, 5, 6, 7]  

1. **Partition pruning** in Spark means reading **only required partitions** based on filter predicates, avoiding full table scans.
2. Spark’s **Catalyst optimizer** analyzes `WHERE` conditions on partition columns (for example `event_date`) at query planning time.
3. Only matching partition paths are sent to **Spark executors (JVM)**, reducing S3/HDFS I/O and task count.
4. Effective pruning requires **explicit filters on partition columns**, not derived expressions.
5. In AWS Glue, partition metadata from the **Glue Data Catalog** enables Spark to skip non-relevant partitions automatically.

1. **AWS Glue Data Catalog** stores table schemas and partition key–to–S3 path mappings used by Spark during query planning.
2. Spark’s **Catalyst optimizer** consults the catalog to identify only the S3 partitions matching filter predicates.
3. As a result, **Spark executors (JVM)** read only relevant partition paths, skipping unnecessary data scans.



How it works 

• Identify partitions: When a query includes a filter on the partition key (e.g., ), the system knows exactly which partitions (folders/files) contain relevant data. 
• Eliminate irrelevant partitions: The optimizer excludes partitions that don't match the filter criteria from the execution plan, saving time and resources. 
• Example: A table partitioned by year with a query  will only scan partitions for 2023 and 2024, skipping older data. [1, 3, 6, 7, 8]  

Types of pruning 

• Static Pruning: Occurs at compile time (e.g., ). 
• Dynamic Pruning: Occurs at run time, often during joins, allowing filtering based on values derived from other tables or complex conditions. [4, 5, 9]  

Benefits 

• Reduced I/O: Less data read from disk. 
• Faster queries: Significantly improves performance, especially for analytical workloads. 
• Lower costs: In cloud systems, fewer bytes scanned mean lower query costs. [1, 3, 5, 7]  

This video explains how partition pruning works in a Spark environment: 
When it works best 

• When tables are partitioned on columns frequently used in  clauses (like dates, regions). 
• When filters are specific enough to narrow down to a few partitions. [4, 6]  

When it might not work 

• Queries that don't filter on partition keys (e.g., ). 
• Complex filters or functions on partition columns can sometimes prevent pruning. 
• Certain external tables in some systems may not support it. [4, 7, 10, 11, 12]  

AI responses may include mistakes.

[1] https://www.dremio.com/wiki/data-partition-pruning/
[2] https://www.enterprisedb.com/docs/epas/latest/application_programming/epas_compat_table_partitioning/03_using_partition_pruning/
[3] https://www.youtube.com/watch?v=m00kyPCqyPE
[4] https://dzone.com/articles/oracle-partition-pruning
[5] https://docs.oracle.com/en/database/oracle/oracle-database/21/vldbg/partition-pruning.html
[6] https://docs.cloudera.com/cdw-runtime/cloud/impala-reference/topics/impala-partition-pruning.html
[7] https://docs.cloud.google.com/bigquery/docs/querying-partitioned-tables
[8] https://docs.oracle.com/cd/E11882_01/server.112/e25523/part_avail.htm
[9] https://www.youtube.com/watch?v=s8Z8Gex7VFw
[10] https://www.youtube.com/watch?v=55wcoi10Wfg
[11] https://www.rapydo.io/blog/sharding-and-partitioning-strategies-in-sql-databases
[12] https://www.baeldung.com/springboot-hibernate-partitionkey-guide

