Window functions in PySpark are powerful operations that enable performing calculations across a set of rows related to the current row. Unlike aggregate functions (which return a single result for a group), **window functions** compute a result for every row while considering a specified "window" of rows. They are widely used in tasks like ranking, cumulative calculations, and comparisons within partitions of data.

---

### **Key Components of a Window Function**
To use a window function in PySpark, you need to define a **Window Specification** and then apply the desired function.

1. **Window Specification**
   The window specification determines:
   - How rows are grouped (partitioning).
   - How rows are ordered within each group.
   - The "frame" of rows to include in calculations (optional).

2. **Window Functions**
   These are operations applied to the defined window, such as:
   - Aggregate functions: `sum`, `avg`, `count`, etc.
   - Ranking functions: `rank`, `dense_rank`, `row_number`.
   - Analytical functions: `lag`, `lead`, `first`, `last`, etc.

---

### **Syntax**
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import <function>

window_spec = Window.partitionBy("column1").orderBy("column2")

df.withColumn("new_column", <function>("column").over(window_spec))
```

---

### **Examples**

#### **1. Ranking Rows**
Suppose you have a dataset of sales by salesperson:

```python
data = [
    ("Alice", "2025-01-01", 300),
    ("Bob", "2025-01-01", 400),
    ("Alice", "2025-01-02", 200),
    ("Bob", "2025-01-02", 500),
]
columns = ["Salesperson", "Date", "Sales"]

df = spark.createDataFrame(data, columns)
```

You want to rank sales within each salesperson group, ordered by the sales amount.

```python
from pyspark.sql.functions import rank
from pyspark.sql.window import Window

# Define window specification
window_spec = Window.partitionBy("Salesperson").orderBy("Sales")

# Apply rank
df_ranked = df.withColumn("Rank", rank().over(window_spec))
df_ranked.show()
```

**Output:**
```
+-----------+----------+-----+----+
|Salesperson|      Date|Sales|Rank|
+-----------+----------+-----+----+
|      Alice|2025-01-02|  200|   1|
|      Alice|2025-01-01|  300|   2|
|        Bob|2025-01-01|  400|   1|
|        Bob|2025-01-02|  500|   2|
+-----------+----------+-----+----+
```

---

#### **2. Cumulative Sum**
You can calculate the cumulative sales for each salesperson over time.

```python
from pyspark.sql.functions import sum

# Define window specification
window_spec_cumsum = Window.partitionBy("Salesperson").orderBy("Date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Calculate cumulative sum
df_cumsum = df.withColumn("CumulativeSales", sum("Sales").over(window_spec_cumsum))
df_cumsum.show()
```

**Output:**
```
+-----------+----------+-----+---------------+
|Salesperson|      Date|Sales|CumulativeSales|
+-----------+----------+-----+---------------+
|      Alice|2025-01-01|  300|            300|
|      Alice|2025-01-02|  200|            500|
|        Bob|2025-01-01|  400|            400|
|        Bob|2025-01-02|  500|            900|
+-----------+----------+-----+---------------+
```

---

#### **3. Row Number**
You can assign a unique row number to each row within a partition.

```python
from pyspark.sql.functions import row_number

# Define window specification
window_spec_row = Window.partitionBy("Salesperson").orderBy("Date")

# Add row number
df_row_number = df.withColumn("RowNumber", row_number().over(window_spec_row))
df_row_number.show()
```

**Output:**
```
+-----------+----------+-----+----------+
|Salesperson|      Date|Sales|RowNumber |
+-----------+----------+-----+----------+
|      Alice|2025-01-01|  300|         1|
|      Alice|2025-01-02|  200|         2|
|        Bob|2025-01-01|  400|         1|
|        Bob|2025-01-02|  500|         2|
+-----------+----------+-----+----------+
```

---

#### **4. Lag and Lead**
You can compare values between rows using `lag` and `lead`.

```python
from pyspark.sql.functions import lag, lead

# Define window specification
window_spec_lag = Window.partitionBy("Salesperson").orderBy("Date")

# Apply lag and lead
df_lag_lead = df.withColumn("PreviousSales", lag("Sales", 1).over(window_spec_lag)) \
                .withColumn("NextSales", lead("Sales", 1).over(window_spec_lag))
df_lag_lead.show()
```

**Output:**
```
+-----------+----------+-----+-------------+---------+
|Salesperson|      Date|Sales|PreviousSales|NextSales|
+-----------+----------+-----+-------------+---------+
|      Alice|2025-01-01|  300|         null|      200|
|      Alice|2025-01-02|  200|          300|     null|
|        Bob|2025-01-01|  400|         null|      500|
|        Bob|2025-01-02|  500|          400|     null|
+-----------+----------+-----+-------------+---------+
```

---

### **Window Specification Details**
1. **Partitioning (`partitionBy`)**
   - Splits the data into groups (like SQL's `GROUP BY`).
   - Each partition operates independently.

2. **Ordering (`orderBy`)**
   - Specifies the order of rows within each partition.

3. **Framing**
   - Controls the range of rows to include in the window for the calculation.
   - Common options:
     - `Window.unboundedPreceding`: Includes all rows from the start.
     - `Window.currentRow`: Includes rows up to the current one.

---

### **Common Window Functions**
- **Ranking Functions**: `row_number`, `rank`, `dense_rank`.
- **Aggregate Functions**: `sum`, `avg`, `count`, `min`, `max`.
- **Analytical Functions**: `lag`, `lead`, `first`, `last`.

---

### **Key Points**
- Window functions don’t reduce the number of rows in the output.
- They are applied on a logical "window" of data.
- Great for tasks like cumulative calculations, ranking, and sequential comparisons.

Let me know if you'd like more examples or help with specific use cases!