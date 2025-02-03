# **Common Pandas DataFrame Operations with Examples**  

`pandas` is a powerful Python library for data manipulation and analysis. Below are **common DataFrame operations** grouped by category.

---

## **1. Creating a DataFrame**
```python
import pandas as pd

# Creating DataFrame from a dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie'], 
        'Age': [25, 30, 35], 
        'City': ['New York', 'London', 'Paris']}

df = pd.DataFrame(data)
print(df)
```
### **Output:**
```
     Name  Age      City
0   Alice   25  New York
1     Bob   30   London
2  Charlie   35    Paris
```

---

## **2. Viewing Data**
### **Basic Information**
```python
df.head(2)   # First 2 rows
df.tail(2)   # Last 2 rows
df.info()    # Summary of DataFrame
df.describe() # Statistical summary of numerical columns
df.shape     # (rows, columns)
df.columns   # List of column names
df.dtypes    # Data types of each column
```

---

## **3. Selecting Data**
### **Select Columns**
```python
df['Name']  # Select single column
df[['Name', 'Age']]  # Select multiple columns
```

### **Select Rows (Using Indexing)**
```python
df.iloc[0]  # Select first row (index-based)
df.iloc[1:3]  # Select rows from index 1 to 2 (excluding 3)

df.loc[0]   # Select row using label-based index
df.loc[0:1, ['Name', 'City']]  # Select specific rows & columns
```

---

## **4. Filtering Data**
### **Using Conditions**
```python
df[df['Age'] > 28]  # Rows where Age > 28
df[df['City'] == 'London']  # Rows where City is London
```

### **Multiple Conditions**
```python
df[(df['Age'] > 28) & (df['City'] == 'London')]  # AND condition
df[(df['Age'] > 28) | (df['City'] == 'London')]  # OR condition
```

---

## **5. Adding & Modifying Columns**
### **Add a New Column**
```python
df['Salary'] = [50000, 60000, 70000]  # Adding a new column
```

### **Modify Existing Column**
```python
df['Age'] = df['Age'] + 1  # Increase Age by 1
```

---

## **6. Removing Data**
```python
df.drop(columns=['Salary'], inplace=True)  # Remove column
df.drop(index=1, inplace=True)  # Remove row with index 1
```

---

## **7. Sorting Data**
```python
df.sort_values(by='Age')  # Sort by Age (ascending)
df.sort_values(by='Age', ascending=False)  # Sort by Age (descending)
```

---

## **8. Handling Missing Values**
```python
df.isnull().sum()  # Count missing values
df.fillna(value="Unknown")  # Fill missing values with "Unknown"
df.dropna()  # Remove rows with missing values
```

---

## **9. Grouping and Aggregation**
```python
df.groupby('City')['Age'].mean()  # Average Age per City
df.groupby('City').agg({'Age': 'max', 'Salary': 'sum'})  # Multiple aggregations
```

---

## **10. Merging and Joining**
### **Concatenation (Stacking DataFrames)**
```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df_concat = pd.concat([df1, df2])  # Stack DataFrames vertically
```

### **Merging DataFrames**
```python
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Salary': [50000, 60000]})

df_merged = pd.merge(df1, df2, on='ID')  # Merge on common column
```

---

## **11. Pivot Tables**
```python
df.pivot_table(index='City', values='Age', aggfunc='mean')
```

---

## **12. Exporting & Importing Data**
```python
df.to_csv('data.csv', index=False)  # Export to CSV
df.to_excel('data.xlsx', index=False)  # Export to Excel

df_csv = pd.read_csv('data.csv')  # Read CSV
df_excel = pd.read_excel('data.xlsx')  # Read Excel
```

---

## **Conclusion**
These commands cover **the most common Pandas operations** used in **data analysis and manipulation**. Let me know if you need **more advanced Pandas functions!**