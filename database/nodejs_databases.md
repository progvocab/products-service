There are very few databases **written entirely in Node.js**, as databases typically require high-performance, low-level system access, which is better suited to **C, C++, Go, or Rust**. However, some **Node.js-based databases** exist, especially for **in-memory processing, lightweight storage, and embedded NoSQL use cases**.

---

## **🔹 Databases Written in Node.js**
### **1️⃣ NeDB**  
✔ **Type**: NoSQL (Document Store, Embedded, Lightweight)  
✔ **Use Case**: In-memory or file-based storage, like a lightweight MongoDB  
✔ **Features**:  
   - **JSON-based document storage**  
   - **MongoDB-like API**  
   - **Fast in-memory operations**  
✔ **Example Use Case**: Small web apps, IoT, local data storage  
✔ **GitHub**: [github.com/louischatriot/nedb](https://github.com/louischatriot/nedb)  

#### **Usage Example (NeDB)**
```javascript
const Datastore = require('nedb');
const db = new Datastore({ filename: 'data.db', autoload: true });

db.insert({ name: 'Alice', age: 25 }, (err, newDoc) => {
    console.log(newDoc);
});
```

---

### **2️⃣ LokiJS**  
✔ **Type**: NoSQL (Document-Oriented, In-Memory)  
✔ **Use Case**: High-speed, in-memory storage with indexing  
✔ **Features**:  
   - **Fast performance** (optimized for JavaScript apps)  
   - **Persistence with optional file storage**  
   - **Event-driven data changes**  
✔ **Example Use Case**: **Single-page apps (SPAs), browser-based storage**  
✔ **GitHub**: [github.com/techfort/LokiJS](https://github.com/techfort/LokiJS)  

#### **Usage Example (LokiJS)**
```javascript
const loki = require('lokijs');
const db = new loki('example.db');

const users = db.addCollection('users');
users.insert({ name: 'Bob', age: 30 });

console.log(users.find({ name: 'Bob' }));
```

---

### **3️⃣ AlaSQL**  
✔ **Type**: SQL + NoSQL Hybrid (Runs in Browser & Node.js)  
✔ **Use Case**: **Client-side SQL processing** (can work without a backend)  
✔ **Features**:  
   - **Supports SQL queries in Node.js & browser**  
   - **Works with CSV, JSON, localStorage**  
   - **Fast in-memory data processing**  
✔ **Example Use Case**: **Handling data in web apps without a full database**  
✔ **GitHub**: [github.com/agershun/alasql](https://github.com/agershun/alasql)  

#### **Usage Example (AlaSQL)**
```javascript
const alasql = require('alasql');

alasql('CREATE TABLE test (id INT, name STRING)');
alasql('INSERT INTO test VALUES (1, "Alice"), (2, "Bob")');

console.log(alasql('SELECT * FROM test')); // Output: [{id: 1, name: "Alice"}, {id: 2, name: "Bob"}]
```

---

### **4️⃣ RxDB**  
✔ **Type**: NoSQL (Reactive, Syncable Database)  
✔ **Use Case**: **Offline-first applications**  
✔ **Features**:  
   - **Uses RxJS for real-time reactivity**  
   - **Syncs with CouchDB, PouchDB**  
   - **Works in browsers, Node.js, and mobile apps**  
✔ **Example Use Case**: **Progressive Web Apps (PWAs), offline applications**  
✔ **GitHub**: [github.com/pubkey/rxdb](https://github.com/pubkey/rxdb)  

#### **Usage Example (RxDB)**
```javascript
import { createRxDatabase, addRxPlugin } from 'rxdb';
import { getRxStorageMemory } from 'rxdb/plugins/storage-memory';

async function run() {
    const db = await createRxDatabase({
        name: 'mydatabase',
        storage: getRxStorageMemory()
    });

    const users = await db.addCollections({
        users: { schema: { title: 'User', version: 0, type: 'object', properties: { name: { type: 'string' } } } }
    });

    await users.users.insert({ name: 'Alice' });
    console.log(await users.users.find().exec());
}

run();
```

---

## **🔹 Popular Databases with Native Node.js Drivers**
Though not written in Node.js, the following databases have strong **native drivers**:

| **Database** | **Type** | **Node.js Driver** |
|-------------|---------|------------------|
| **MongoDB** | NoSQL (Document) | `mongodb` |
| **PostgreSQL** | SQL (Relational) | `pg` |
| **MySQL** | SQL (Relational) | `mysql2` |
| **Redis** | NoSQL (Key-Value) | `ioredis` |
| **Cassandra** | NoSQL (Column Store) | `cassandra-driver` |
| **Firebase Firestore** | NoSQL (Realtime DB) | `firebase-admin` |
| **DynamoDB** | NoSQL (AWS Key-Value) | `aws-sdk` |

---

## **🚀 Summary: Which Node.js Database Should You Use?**
| **Database** | **Best For** |
|-------------|-------------|
| **NeDB** | Small JSON-based storage, like lightweight MongoDB |
| **LokiJS** | Fast in-memory NoSQL, for single-page apps |
| **AlaSQL** | SQL-like queries in browser/Node.js (CSV, JSON) |
| **RxDB** | Offline-first apps with real-time sync |

Node.js is typically used **as a database client** rather than for database engines because **low-level performance optimization** is better suited for **C, C++, Rust, or Go**.

Would you like recommendations based on **your specific use case**?