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