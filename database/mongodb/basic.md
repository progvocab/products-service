Here are some **simple MongoDB queries** for common operations using the `mongo` shell or MongoDB Compass:

---

### **1. Insert a Document**

```js
db.users.insertOne({ name: "Alice", age: 25, city: "New York" });
```

Insert multiple:
```js
db.users.insertMany([
  { name: "Bob", age: 30 },
  { name: "Charlie", age: 22 }
]);
```

---

### **2. Find Documents**

Find all:
```js
db.users.find();
```

Find with condition:
```js
db.users.find({ age: 25 });
```

Find with projection:
```js
db.users.find({ age: { $gt: 20 } }, { name: 1, _id: 0 });
```

---

### **3. Update Documents**

Update one:
```js
db.users.updateOne({ name: "Alice" }, { $set: { city: "Boston" } });
```

Update many:
```js
db.users.updateMany({ age: { $lt: 30 } }, { $inc: { age: 1 } });
```

---

### **4. Delete Documents**

Delete one:
```js
db.users.deleteOne({ name: "Bob" });
```

Delete many:
```js
db.users.deleteMany({ age: { $gt: 40 } });
```

---

### **5. Sort and Limit**

```js
db.users.find().sort({ age: -1 }).limit(2);
```

---

### **6. Count Documents**

```js
db.users.countDocuments({ city: "New York" });
```

---

### **7. Aggregation (Basic Example)**

Group by city and count users:
```js
db.users.aggregate([
  { $group: { _id: "$city", total: { $sum: 1 } } }
]);
```

To calculate the **sum of salary for all employees** in MongoDB, you can use the **aggregation framework**:

---

### **MongoDB Query**

```javascript
db.employees.aggregate([
  {
    $group: {
      _id: null,
      totalSalary: { $sum: "$salary" }
    }
  }
]);
```

---

### **Explanation**

- `$group`: Groups all documents into one group (since `_id: null`).
- `$sum: "$salary"`: Adds up all the values in the `salary` field.

---

### **Sample Data**

```json
[
  { "name": "Alice", "salary": 5000 },
  { "name": "Bob", "salary": 7000 },
  { "name": "Charlie", "salary": 6000 }
]
```

---

### **Result**

```json
[
  { "_id": null, "totalSalary": 18000 }
]
```

---

Would you like the same aggregation using **Spring Data MongoDB** or Java driver code?


---

Would you like examples with Spring Data MongoDB as well?


Sure! Let's explore more advanced MongoDB **update queries** with options like `upsert`, `multi` (now handled by `updateMany()`), and a query that **increments a field by 10%**.

---

### **1. `updateOne()` with `upsert: true`**

This will **update** the document if it exists, otherwise **insert** a new one.

```js
db.products.updateOne(
  { sku: "A1001" },
  { $set: { price: 50, stock: 100 } },
  { upsert: true }
);
```

---

### **2. `updateMany()` with `$inc`, `$set`**

Update **multiple** documents (formerly required `multi: true`):

```js
db.products.updateMany(
  { category: "electronics" },
  {
    $inc: { stock: 5 },
    $set: { updated: true }
  }
);
```

---

### **3. Increment a numeric field by 10%**

Use `$mul` operator:

```js
db.products.updateMany(
  { category: "electronics" },
  { $mul: { price: 1.10 } }
);
```

> This increases the price by 10%. `$mul` is available from MongoDB 4.2 onward.

---

### **Optional: Combined Example with All**

```js
db.products.updateOne(
  { sku: "A2001" },
  {
    $set: { category: "accessories" },
    $inc: { stock: 10 },
    $mul: { price: 1.10 }
  },
  { upsert: true }
);
```

---

Let me know if you want Spring Data MongoDB equivalents or examples using aggregation pipelines!


Awesome! Here’s a complete set of **Spring Data MongoDB examples** for both **basic and advanced queries**, along with **initial data** and **expected results**.

---

## **Sample Entity**

```java
@Document("products")
public class Product {
    @Id
    private String id;
    private String sku;
    private String name;
    private String category;
    private double price;
    private int stock;
    private boolean updated;
}
```

---

## **1. Insert Documents**

### **Initial Data:**

```java
repository.saveAll(List.of(
    new Product("1", "A1001", "Phone", "electronics", 500.0, 50, false),
    new Product("2", "A1002", "Laptop", "electronics", 1000.0, 30, false),
    new Product("3", "A1003", "Charger", "accessories", 20.0, 100, false)
));
```

---

## **2. Find All**

```java
List<Product> products = repository.findAll();
```

### **Result:**
Returns all 3 products.

---

## **3. Find with Condition and Projection**

```java
List<Product> cheapProducts = repository.findByPriceLessThan(100.0);
```

You can use a **Projection** interface if needed.

---

## **4. UpdateOne with Upsert**

```java
Query query = new Query(Criteria.where("sku").is("A2001"));
Update update = new Update()
        .set("price", 50)
        .set("stock", 100)
        .set("category", "accessories");
mongoTemplate.upsert(query, update, Product.class);
```

### **Result:**
If `sku: A2001` didn’t exist, it's now inserted.

---

## **5. UpdateMany with `$inc` and `$set`**

```java
Query query = new Query(Criteria.where("category").is("electronics"));
Update update = new Update()
        .inc("stock", 5)
        .set("updated", true);
mongoTemplate.updateMulti(query, update, Product.class);
```

### **Result:**

| SKU   | Stock Before | Stock After | Updated |
|--------|--------------|-------------|---------|
| A1001 | 50           | 55          | true    |
| A1002 | 30           | 35          | true    |

---

## **6. Increment by 10% using `$mul`**

```java
Query query = new Query(Criteria.where("category").is("electronics"));
Update update = new Update().multiply("price", 1.10);
mongoTemplate.updateMulti(query, update, Product.class);
```

> Note: `multiply()` requires MongoDB **4.2+**.

### **Result:**

| SKU   | Price Before | Price After |
|--------|--------------|-------------|
| A1001 | 500.0        | 550.0       |
| A1002 | 1000.0       | 1100.0      |

---

## **7. Combined Upsert with `$set`, `$inc`, `$mul`**

```java
Query query = new Query(Criteria.where("sku").is("A3001"));
Update update = new Update()
        .set("category", "gadgets")
        .inc("stock", 10)
        .multiply("price", 1.10);
mongoTemplate.upsert(query, update, Product.class);
```

### **Result:**

New document created with:
```json
{
  "sku": "A3001",
  "category": "gadgets",
  "stock": 10,
  "price": 0.0  // price will remain 0 unless initialized
}
```

---

Would you like a sample repository interface (`ProductRepository`) and REST controller too?


Absolutely! Below are MongoDB **aggregation queries** for your employee-related use cases, followed by their **Spring Data MongoDB equivalents**, along with sample **input data** and **expected results**.

---

## **Sample Employee Document**

```json
{
  "_id": ObjectId("..."),
  "name": "Alice",
  "department": "Engineering",
  "salary": 70000
}
```

---

## **1. Total Number of Employees in Each Department**

### **MongoDB Query**

```js
db.employees.aggregate([
  { $group: { _id: "$department", total: { $sum: 1 } } }
]);
```

### **Spring Data MongoDB**

```java
Aggregation agg = Aggregation.newAggregation(
    Aggregation.group("department").count().as("total")
);

AggregationResults<Document> results = mongoTemplate.aggregate(agg, "employees", Document.class);
```

---

## **2. Employee(s) with Highest Salary**

### **MongoDB Query**

```js
db.employees.find().sort({ salary: -1 }).limit(1);
```

### **Spring Data MongoDB**

```java
Query query = new Query().with(Sort.by(Sort.Direction.DESC, "salary")).limit(1);
Employee topPaid = mongoTemplate.findOne(query, Employee.class);
```

---

## **3. Employee(s) with Second Highest Salary**

### **MongoDB Aggregation**

```js
db.employees.aggregate([
  { $sort: { salary: -1 } },
  { $group: { _id: "$salary", employees: { $push: "$$ROOT" } } },
  { $skip: 1 },
  { $limit: 1 },
  { $unwind: "$employees" },
  { $replaceRoot: { newRoot: "$employees" } }
]);
```

### **Spring Data MongoDB**

Spring Data doesn't directly support `skip + group + unwind` in aggregation pipeline easily, but this is a close approximation using raw `Document` parsing:

```java
Aggregation agg = Aggregation.newAggregation(
    Aggregation.sort(Sort.by(Sort.Direction.DESC, "salary")),
    Aggregation.group("salary").push("$$ROOT").as("employees"),
    Aggregation.skip(1),
    Aggregation.limit(1),
    Aggregation.unwind("employees"),
    Aggregation.replaceRoot("employees")
);

AggregationResults<Employee> result = mongoTemplate.aggregate(agg, "employees", Employee.class);
```

---

## **4. Maximum Salary for Each Department**

### **MongoDB Aggregation**

```js
db.employees.aggregate([
  { $group: { _id: "$department", maxSalary: { $max: "$salary" } } }
]);
```

### **Spring Data MongoDB**

```java
Aggregation agg = Aggregation.newAggregation(
    Aggregation.group("department").max("salary").as("maxSalary")
);

AggregationResults<Document> result = mongoTemplate.aggregate(agg, "employees", Document.class);
```

---

## **Sample Input Data**

```json
[
  { "name": "Alice", "department": "Engineering", "salary": 70000 },
  { "name": "Bob", "department": "Engineering", "salary": 80000 },
  { "name": "Charlie", "department": "HR", "salary": 60000 },
  { "name": "Daisy", "department": "Engineering", "salary": 85000 },
  { "name": "Eva", "department": "HR", "salary": 65000 }
]
```

---

## **Expected Results**

| Use Case                       | Output (Example)                                                  |
|--------------------------------|-------------------------------------------------------------------|
| Employees per department       | `{ _id: "Engineering", total: 3 }`, `{ _id: "HR", total: 2 }`     |
| Highest salary                 | `{ name: "Daisy", salary: 85000 }`                                |
| Second highest salary          | `{ name: "Bob", salary: 80000 }`                                  |
| Max salary per department      | `{ _id: "Engineering", maxSalary: 85000 }`, `{ _id: "HR", maxSalary: 65000 }` |

---

Would you also like versions of these as **Spring Data JPA (SQL)** queries for comparison?