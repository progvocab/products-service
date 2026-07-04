### HNSW

**HNSW (Hierarchical Navigable Small World)** is one of the most popular **Approximate Nearest Neighbor (ANN)** algorithms used in vector databases for **fast semantic search**. In a RAG (Retrieval-Augmented Generation) system, it helps retrieve the most relevant document embeddings without comparing the query against every vector.

Instead of performing an exhaustive search over millions of embeddings, HNSW organizes vectors into a graph so that nearest neighbors can be found efficiently.

 

Suppose you have:

* 100 million document chunks
* Each represented by a 768-dimensional embedding

A user asks:

> "How does gradient accumulation work?"

The query is converted into an embedding, and you need to find the most similar document vectors.

A brute-force search would compute the similarity with **all 100 million vectors**, which is too slow for interactive applications.

HNSW reduces this search from a linear scan to a graph traversal that visits only a tiny fraction of the vectors.

 

###  RAG pipeline
- Ingestion

```text
                    Documents
                        │
                        ▼
               Chunk Documents
                        │
                        ▼
              Generate Embeddings
                        │
                        ▼
              Build HNSW Index
                        │
                        ▼
                Store in Vector DB
```
- Retrieval                  
```text
                     
                    User Query
                        │
                        ▼
              Generate Query Embedding
                        │
                        ▼
              Search HNSW Graph
                        │
                        ▼
              Top-K Similar Chunks
                        │
                        ▼
                 Build Prompt
                        │
                        ▼
                      LLM
```



### Problem HNSW solves 

Imagine these document embeddings:

```text
A

B

C

D

E

F

G

H
```

A brute-force search computes:

```text
Query

↓

Compare with A

Compare with B

Compare with C

...

Compare with H
```

Time complexity:

**O(N)**



HNSW builds connections between nearby vectors.

```text
        A ------ B ------- C

       /          \         \

      D --------- E -------- F

       \          /          /

        G ------- H ---------
```

Instead of checking every node:

```text
Query

↓

Jump near E

↓

Visit F

↓

Visit C

↓

Found nearest neighbors
```

Only a small subset of nodes is explored.
 

### Why is it called "Hierarchical"?

HNSW has multiple graph layers.

Top layers contain only a few nodes.

Lower layers contain progressively more nodes.

Example:

```text
Layer 3

        A

        |

Layer 2

     B ----- C

      \     /

Layer 1

 D --- E --- F

 |     |     |

Layer 0

A B C D E F G H I J K L
```

The search starts at the sparse top layer and descends through denser layers until it reaches the bottom layer containing all vectors.

This is similar to navigating:

* Country
* State
* City
* Street
* House

Rather than checking every house in the country.

 

### Building the HNSW graph

Suppose vectors arrive one by one.

Insert A

```text
A
```

Insert B

```text
A ----- B
```

Insert C

```text
A ----- B ----- C
```

Insert D

```text
A ----- B ----- C

 \

  D
```

Each new vector is connected to its nearest existing neighbors.

Over time the graph becomes highly connected.

 

### Searching the graph

Suppose the query is near **F**.

Search begins at the top layer.

```text
Start

↓

A

↓

Closer?

↓

Move to B

↓

Move to E

↓

Move to F

↓

Done
```

Rather than exploring every node, the algorithm greedily moves toward increasingly similar neighbors.

---

# Why is it called "Small World"?

HNSW is based on the **small-world network** idea, where most nodes are connected through surprisingly short paths.

Examples include:

* Social networks
* Airline route maps
* The web

You can often reach a distant node in only a few hops.

HNSW applies this principle to embeddings.

 

### Important parameters

#### 1. M

Number of neighbors stored for each node.

Example:

```
M = 4
```

Each vector connects to at most four nearby vectors.

Larger **M** gives:

* Better recall
* More memory usage
* Longer index construction time



#### 2. efConstruction

Controls the amount of work during index building.

Higher values mean:

* Better graph quality
* Slower index creation

Typical values:

```
100

200

400
```



#### 3. efSearch

Controls how many candidate nodes are explored during search.

Higher values mean:

* Higher recall
* Higher latency

Typical values:

```
50

100

300
```



### Time complexity

| Operation          | Complexity               |
| ------------------ | ------------------------ |
| Brute-force search | O(N)                     |
| HNSW search        | Approximately O(log N)   |
| Index construction | Approximately O(N log N) |

Although not guaranteed mathematically, HNSW typically behaves close to logarithmic time in practice.




Imagine:

```
10 Million vectors
```

Brute force:

```
10,000,000 similarity calculations
```

HNSW:

```
≈ Hundreds to a few thousand graph traversals
```

This is why vector databases can return results in milliseconds.



### Advantages

* Extremely fast search
* High recall (often over 95%)
* Scales to millions or billions of vectors
* Widely used and well supported
* Excellent performance for high-dimensional embeddings



### Disadvantages

* Requires additional memory for graph links
* Index creation can be time-consuming
* Updates are more expensive than simple array inserts
* Not ideal for rapidly changing datasets with frequent deletions



### Other ANN Algorithms

| Algorithm                 | Search Speed | Recall      |          Memory Usage | Dynamic Updates | Typical Use                 |
| ------------------------- | -----------: | ----------- | --------------------: | --------------- | --------------------------- |
| Brute Force               |         Slow | 100%        |                   Low | Excellent       | Small datasets              |
| HNSW                      |    Very Fast | Very High   |                  High | Good            | General-purpose RAG         |
| IVF (Inverted File Index) |         Fast | Medium–High |                Medium | Good            | Very large datasets         |
| Product Quantization (PQ) |    Very Fast | Medium      |              Very Low | Good            | Memory-constrained systems  |
| IVF + PQ                  |    Very Fast | Medium–High |                   Low | Good            | Billion-scale vector search |
| DiskANN                   |    Very Fast | Very High   | Low RAM (disk-backed) | Good            | Massive datasets on SSDs    |



###  Vector Databases

Many vector databases either use HNSW directly or offer it as one of their indexing options:

| Vector Database | HNSW Support                |
| --------------- | --------------------------- |
| FAISS           | Yes                         |
| Milvus          | Yes                         |
| Qdrant          | Yes                         |
| Weaviate        | Yes                         |
| OpenSearch      | Yes (k-NN plugin)           |
| Elasticsearch   | Yes (dense vector indexing) |
| Chroma          | Yes (depending on backend)  |



HNSW is the **indexing structure** that makes semantic retrieval practical in RAG systems. After documents are converted into embeddings, HNSW organizes those vectors into a hierarchical graph of nearest neighbors. When a user submits a query, its embedding is generated and the graph is traversed from coarse layers to fine layers, quickly locating the most similar document chunks. This approach avoids scanning every vector, delivering **low-latency, high-recall retrieval** that is essential for production-scale RAG applications with millions of embeddings.
