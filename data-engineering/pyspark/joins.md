In PySpark, joins are used to combine two datasets (DataFrames) based on a condition, typically involving matching keys. PySpark supports several types of joins, each serving different purposes. Here’s an overview of the join types:

---

### **1. Inner Join**
- Returns only the rows where the keys match in both DataFrames.
- Default join type in PySpark.

#### **Code Example**
```python
df1.join(df2, df1.id == df2.id, "inner").show()
```

#### **Use Case**
When you want to retrieve records that exist in both DataFrames.

---

### **2. Left Outer Join (Left Join)**
- Returns all rows from the left DataFrame and the matching rows from the right DataFrame.
- If no match is found, `null` values are returned for columns from the right DataFrame.

#### **Code Example**
```python
df1.join(df2, df1.id == df2.id, "left").show()
```

#### **Use Case**
When you want to keep all data from the left DataFrame and only the matched rows from the right.

---

### **3. Right Outer Join (Right Join)**
- Returns all rows from the right DataFrame and the matching rows from the left DataFrame.
- If no match is found, `null` values are returned for columns from the left DataFrame.

#### **Code Example**
```python
df1.join(df2, df1.id == df2.id, "right").show()
```

#### **Use Case**
When you want to keep all data from the right DataFrame and only the matched rows from the left.

---

### **4. Full Outer Join (Outer Join)**
- Returns all rows when there is a match in either DataFrame.
- Unmatched rows from both DataFrames will have `null` values for the missing columns.

#### **Code Example**
```python
df1.join(df2, df1.id == df2.id, "outer").show()
```

#### **Use Case**
When you want to combine all rows from both DataFrames, regardless of whether they match.

---

### **5. Left Semi Join**
- Returns rows from the left DataFrame that have matching keys in the right DataFrame.
- Does not include columns from the right DataFrame.

#### **Code Example**
```python
df1.join(df2, df1.id == df2.id, "left_semi").show()
```

#### **Use Case**
When you want to filter the left DataFrame based on keys present in the right DataFrame.

---

### **6. Left Anti Join**
- Returns rows from the left DataFrame that **do not** have matching keys in the right DataFrame.

#### **Code Example**
```python
df1.join(df2, df1.id == df2.id, "left_anti").show()
```

#### **Use Case**
When you want to find rows in the left DataFrame that are not present in the right DataFrame.

---

### **7. Cross Join (Cartesian Product)**
- Returns the Cartesian product of the two DataFrames, i.e., every row from the first DataFrame is paired with every row from the second DataFrame.

#### **Code Example**
```python
df1.crossJoin(df2).show()
```

#### **Use Case**
When you want to combine every row of both DataFrames. Use cautiously, as it can generate a large number of rows.

---

### **Comparison of Join Types**

| **Join Type**       | **Description**                                   | **Nulls Introduced** |
|---------------------|---------------------------------------------------|-----------------------|
| Inner Join          | Matches rows in both DataFrames.                  | No                   |
| Left Outer Join     | All rows from the left, matched rows from right.  | Yes                  |
| Right Outer Join    | All rows from the right, matched rows from left.  | Yes                  |
| Full Outer Join     | All rows from both, unmatched rows filled with nulls. | Yes              |
| Left Semi Join       | Filters left DataFrame by matching keys.         | No                   |
| Left Anti Join       | Filters left DataFrame by non-matching keys.     | No                   |
| Cross Join          | Combines all rows from both DataFrames.           | No                   |

---

### **Example with Sample Data**
```python
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("JoinsExample").getOrCreate()

# Create sample DataFrames
data1 = [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
data2 = [(1, "HR"), (3, "Finance"), (4, "IT")]

df1 = spark.createDataFrame(data1, ["id", "name"])
df2 = spark.createDataFrame(data2, ["id", "department"])

# Perform different joins
df1.join(df2, "id", "inner").show()       # Inner Join
df1.join(df2, "id", "left").show()        # Left Join
df1.join(df2, "id", "right").show()       # Right Join
df1.join(df2, "id", "outer").show()       # Full Outer Join
df1.join(df2, "id", "left_semi").show()   # Left Semi Join
df1.join(df2, "id", "left_anti").show()   # Left Anti Join
df1.crossJoin(df2).show()                 # Cross Join
```

Would you like further clarification or help with a specific join?