Here's a comprehensive overview of **basic queries** for an `employee` collection in **MongoDB**, **Spring Data JPA (for relational DBs)**, and **Spring Data MongoTemplate**:

---

### **1. Find Employees**

#### MongoDB Shell:
```js
db.employee.find({ department: "IT" });
```

#### Spring Data JPA:
```java
List<Employee> findByDepartment(String department);
```

#### MongoTemplate:
```java
Query query = new Query(Criteria.where("department").is("IT"));
List<Employee> employees = mongoTemplate.find(query, Employee.class);
```

---

### **2. Update Salary**

#### MongoDB Shell:
```js
db.employee.updateOne(
  { name: "Alice" },
  { $set: { salary: 7000 } }
);
```

#### MongoTemplate:
```java
Query query = new Query(Criteria.where("name").is("Alice"));
Update update = new Update().set("salary", 7000);
mongoTemplate.updateFirst(query, update, Employee.class);
```

> *JPA typically doesn't update specific fields directly—you retrieve, modify, and save.*

---

### **3. Delete Employee**

#### MongoDB Shell:
```js
db.employee.deleteOne({ name: "Alice" });
```

#### Spring Data JPA:
```java
void deleteByName(String name);
```

#### MongoTemplate:
```java
Query query = new Query(Criteria.where("name").is("Alice"));
mongoTemplate.remove(query, Employee.class);
```

---

### **4. Sum of Salary**

#### MongoDB Shell:
```js
db.employee.aggregate([
  { $group: { _id: null, total: { $sum: "$salary" } } }
]);
```

#### MongoTemplate:
```java
Aggregation agg = Aggregation.newAggregation(
    Aggregation.group().sum("salary").as("total")
);
AggregationResults<Document> results = mongoTemplate.aggregate(agg, "employee", Document.class);
```

> Not available out-of-the-box in JPA; would require custom JPQL/native SQL query.

---

### **5. Sort by Salary Descending**

#### MongoDB Shell:
```js
db.employee.find().sort({ salary: -1 });
```

#### JPA:
```java
List<Employee> findAllByOrderBySalaryDesc();
```

#### MongoTemplate:
```java
Query query = new Query().with(Sort.by(Sort.Direction.DESC, "salary"));
List<Employee> employees = mongoTemplate.find(query, Employee.class);
```

---

### **6. Count by Department**

#### MongoDB Shell:
```js
db.employee.aggregate([
  { $group: { _id: "$department", count: { $sum: 1 } } }
]);
```

#### MongoTemplate:
```java
Aggregation agg = Aggregation.newAggregation(
    Aggregation.group("department").count().as("count")
);
mongoTemplate.aggregate(agg, "employee", Document.class);
```

> JPA version requires native or JPQL query for group by and count.

---

Would you like all these in a table format or want to see full class/POJO examples as well?


Here's how to write **custom queries using `@Query` annotations** in **Spring Data JPA** and **Spring Data MongoDB** for typical operations on an `Employee` collection.

---

## **1. Finding Employees by Department**

### **Spring Data JPA**
```java
@Query("SELECT e FROM Employee e WHERE e.department = :dept")
List<Employee> findByDepartment(@Param("dept") String department);
```

### **Spring Data MongoDB**
```java
@Query("{ 'department': ?0 }")
List<Employee> findByDepartment(String department);
```

---

## **2. Updating Salary by Name**

> `@Query` can’t update in Spring Data MongoDB directly — you need to use `MongoTemplate` or `@Modifying` in JPA.

### **Spring Data JPA**
```java
@Modifying
@Query("UPDATE Employee e SET e.salary = :salary WHERE e.name = :name")
int updateSalaryByName(@Param("name") String name, @Param("salary") Double salary);
```

---

## **3. Delete by Name**

### **Spring Data JPA**
```java
@Modifying
@Query("DELETE FROM Employee e WHERE e.name = :name")
void deleteByName(@Param("name") String name);
```

### **Spring Data MongoDB**
```java
@Query(value = "{ 'name': ?0 }", delete = true)
void deleteByName(String name);
```

---

## **4. Count by Department**

> For aggregation queries, use `MongoTemplate` in MongoDB. In JPA, use a native query.

### **Spring Data JPA (Native SQL)**
```java
@Query(value = "SELECT department, COUNT(*) FROM employee GROUP BY department", nativeQuery = true)
List<Object[]> countByDepartment();
```

---

## **5. Sum of Salary**

### **Spring Data JPA**
```java
@Query("SELECT SUM(e.salary) FROM Employee e")
Double getTotalSalary();
```

---

## **6. Sort by Salary Descending**

### **Spring Data JPA**
```java
@Query("SELECT e FROM Employee e ORDER BY e.salary DESC")
List<Employee> findAllSortedBySalaryDesc();
```

---

## **7. Find Employee with Max Salary**

### **Spring Data JPA**
```java
@Query("SELECT e FROM Employee e WHERE e.salary = (SELECT MAX(e2.salary) FROM Employee e2)")
Employee findEmployeeWithMaxSalary();
```

---

## **8. MongoDB Sum with Projection (Spring Data Mongo)**
```java
@Query(value = "{}", fields = "{ 'salary' : 1 }")
List<Employee> findAllSalaries();
```
> For real aggregation (sum), prefer `MongoTemplate`.

---

Let me know if you'd like the full repository interfaces with all these examples, or if you want equivalent queries for Spring MongoTemplate side-by-side.