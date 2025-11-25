# ORC 

ORC (Optimized Row Columnar) is a **columnar storage file format** designed for big-data analytics. It provides **high compression**, **fast predicate pushdown**, **type-aware encoding**, and **splittable column stripes**, making reads and scans efficient.


An ORC file has:

* **Stripes**: Large horizontal chunks (e.g., 64–256 MB) containing rows.
* **Indexes**: Min/max value per column chunk for predicate pushdown.
* **Column Data**: Encoded values (RLE, dictionary, bit-packing).
* **Footer**: Schema, statistics, stripe metadata.
* **Postscript**: Compression info and version.

### Features

* Highly optimized for **analytical workloads**
* Efficient for **columnar reads**, skipping unnecessary columns
* Built-in **compression and statistics**
* Great for **Spark**, **Hive**, **Presto/Trino**, **AWS EMR**
* Ideal for periodic **append-only or batch updates**

### Primary Use Cases

* Data lake storage in Hadoop/S3
* ETL pipelines (Spark, Hive, AWS Glue, EMR)
* Large fact tables requiring predicate pushdown
* High-performance OLAP queries
* Storing tables before querying with Presto/Trino/Athena

```
pip install pyorc
```

Writing an ORC File 

```python
import pyorc

with open("employees.orc", "wb") as f:
    writer = pyorc.Writer(
        f,
        "struct<id:int,name:string,salary:double>"
    )
    writer.write((1, "Alice", 90000.0))
    writer.write((2, "Bob", 85000.0))
    writer.close()
```

 Reading an ORC File

```python
import pyorc

with open("employees.orc", "rb") as f:
    reader = pyorc.Reader(f)
    for row in reader:
        print(row)
```

Using ORC With Pandas (Convert DataFrame to ORC)

Pandas does not directly write ORC, so we use **pyarrow**.

```shell
pip install pyarrow
```

```python
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc

df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 28, 25]
})

table = pa.Table.from_pandas(df)

with open("people.orc", "wb") as f:
    orc.write_table(table, f)
```

Reading ORC With PyArrow

```python
import pyarrow.orc as orc

with open("people.orc", "rb") as f:
    table = orc.ORCFile(f).read()
    df = table.to_pandas()

print(df)
```

### When to Prefer ORC Over Parquet

Use ORC when:

* You use **Hive**, **Presto**, **Trino**, **EMR**
* You need **fast predicate pushdown** on numeric columns
* You need **intensive compression** for large tables
* Schema rarely changes

Parquet may be preferred for streaming or ML pipelines.

More :  also compare **ORC vs Parquet vs Avro** or show **Spark read/write examples**.
