Here’s a **visual and detailed flow of query execution** in Elasticsearch (`server/` module) with explanations of key classes:

---

### **Query Execution Flow (Search Path)**

```
[Client Request]
     |
     V
[RestController]
  -> Maps REST API to internal request
     |
     V
[TransportSearchAction]
  -> Dispatches to coordinating node
     |
     V
[SearchService]
  -> Prepares query context
     |
     V
[QueryPhase]
  -> Builds and executes Lucene query
     |
     V
[Engine -> InternalEngine]
  -> Access Lucene index reader
     |
     V
[IndexShard]
  -> Coordinates per-shard search
     |
     V
[SearchPhaseController]
  -> Collects results from all shards
     |
     V
[Response Sent to Client]
```

---

### **Key Classes & Their Roles**

| Class/File | Description |
|------------|-------------|
| `RestSearchAction` | Handles the REST `/search` endpoint. Converts to `SearchRequest`. |
| `TransportSearchAction` | Handles sending the search request to the correct shards. |
| `SearchService` | Parses the query, prepares the `SearchContext`, executes phases. |
| `QueryPhase` | Executes the actual Lucene query logic. |
| `InternalEngine` | Reads from Lucene segment files. Writes happen here too. |
| `IndexShard` | Coordinates local read/write operations at the shard level. |
| `SearchPhaseController` | Gathers and merges results from multiple shards. |

---

### **Lucene Integration**
- Elasticsearch uses Lucene for actual **document retrieval**.
- Lucene queries are executed via `IndexSearcher`, which is wrapped inside the `Engine`.

---

### **Important Design Choices**
- **Segment-based storage**: Immutable segments allow concurrent reads and background merges.
- **Document-oriented**: JSON docs are indexed and stored with fields broken into tokens.
- **Sharded architecture**: Parallel execution improves performance.
- **Caching**: Uses request and field data caches for faster lookups.

---

### **Example Path for a Query**
```json
GET /employees/_search
{
  "query": {
    "match": {
      "title": "engineer"
    }
  }
}
```

This triggers:
- `RestSearchAction` -> `TransportSearchAction` -> `SearchService`
- Parses query -> builds Lucene Query -> executes via `InternalEngine`
- Hits Lucene’s inverted index -> fetches matching documents

---

Would you like me to **show source file locations** for each step in GitHub or help you set up a debugger to trace this in action?