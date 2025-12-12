 # Mongo DB

1. **Logical View** → How data is organized/modeled (databases, collections, documents, indexes).
2. **Physical View** → How MongoDB is deployed and scaled (replica sets, shards, config servers, query routers).
 


##  Logical Architecture (Data Model)

```mermaid
graph TD
    A[Database] --> B[Collection 1]
    A --> C[Collection 2]
    B --> D[Document 1]
    B --> E[Document 2]
    C --> F[Document 3]
    D --> G[Fields + Values]
    E --> H[Embedded Documents]
    F --> I[Indexes]
```

 Explanation:

* **Database**: A container of collections.
* **Collection**: Group of documents (like tables in RDBMS).
* **Document**: JSON-like object (BSON internally).
* **Fields**: Key-value pairs.
* **Indexes**: Improve query performance.

 

## Physical Architecture (Deployment Model)

```mermaid
graph TD
    subgraph Shard1
        RS1P[Primary Replica] --> RS1S1[Secondary Replica]
        RS1P --> RS1S2[Secondary Replica]
    end
    
    subgraph Shard2
        RS2P[Primary Replica] --> RS2S1[Secondary Replica]
        RS2P --> RS2S2[Secondary Replica]
    end
    
    subgraph ConfigServers
        C1[Config Server 1]
        C2[Config Server 2]
        C3[Config Server 3]
    end
    
    subgraph Routers
        M1[MongoS Router 1]
        M2[MongoS Router 2]
    end
    
    Client --> M1
    Client --> M2
    M1 --> RS1P
    M1 --> RS2P
    M2 --> RS1P
    M2 --> RS2P
    M1 --> C1
    M2 --> C2
```


[Node and Shard](Nodes.md)
* **Replica Set**: Primary + Secondaries (HA + failover).
* **Sharding**: Distributes data across shards.
* **Config Servers**: Store metadata for sharding.
* **MongoS Routers**: Clients connect here; they route queries to the right shard.

 

### Logical + Physical Architecture

how a query from the **client** flows through **MongoS**, gets routed to the correct **shard**, and finally resolves to a **collection/document** inside the logical database.
 

```mermaid
graph TD

    subgraph Client Layer
        U[Client Application]
    end

    subgraph Router Layer
        R1[MongoS Router 1]
        R2[MongoS Router 2]
    end

    subgraph Config Servers
        CS1[Config Server 1]
        CS2[Config Server 2]
        CS3[Config Server 3]
    end

    subgraph Shard1
        RS1P[Primary Replica]
        RS1S1[Secondary Replica]
        RS1S2[Secondary Replica]
    end

    subgraph Shard2
        RS2P[Primary Replica]
        RS2S1[Secondary Replica]
        RS2S2[Secondary Replica]
    end

    subgraph Logical View
        DB[Database]
        C1[Collection]
        C2[Collection]
        DOC1[Document]
        DOC2[Document]
        IDX[Index]
    end

    %% Connections
    U --> R1
    U --> R2

    R1 --> CS1
    R2 --> CS2

    R1 --> RS1P
    R1 --> RS2P
    R2 --> RS1P
    R2 --> RS2P

    RS1P --> DB
    RS2P --> DB

    DB --> C1
    DB --> C2
    C1 --> DOC1
    C1 --> DOC2
    C2 --> IDX
```

- [CAP](CAP.md)
  MongoDB acts as AP , prioritize availability over correctness



### Operations 
- [Basic Queries](basic.md)
- [Map Reduce Queries](MapReduce.md)
 

1. **Client Layer** → Applications talk to MongoDB through **MongoS routers**.
2. **Router Layer (MongoS)** → Handles query routing, talks to config servers for shard metadata.
3. **Config Servers** → Store cluster metadata (shard mappings).
4. **Shards** → Each shard is a **replica set** (HA, scaling).
5. **Logical View inside Shards** → Each replica set stores **databases → collections → documents → indexes**.
 


