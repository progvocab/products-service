

# HashMap

### UML 
```mermaid
classDiagram
    class HashMap~K,V~ {
        -int size
        -int threshold
        -float loadFactor
        -int modCount
        -Node~K,V~[] table
        -Set~Map.Entry~<K,V>~> entrySet
        +HashMap()
        +HashMap(int initialCapacity)
        +HashMap(int initialCapacity, float loadFactor)
        +V get(Object key)
        +V put(K key, V value)
        +V remove(Object key)
        +void clear()
        +boolean containsKey(Object key)
        +boolean containsValue(Object value)
        +int size()
        +Set~K~ keySet()
        +Collection~V~ values()
        +Set~Map.Entry~<K,V>~> entrySet()
        -int hash(Object key)
        -Node~K,V~ getNode(int hash, Object key)
        -V putVal(int hash, K key, V value, boolean onlyIfAbsent, boolean evict)
    }

    class Node~K,V~ {
        +int hash
        +K key
        +V value
        +Node~K,V~ next
        +K getKey()
        +V getValue()
        +V setValue(V value)
        +boolean equals(Object o)
        +int hashCode()
    }

    class EntrySet {
        +Iterator~Map.Entry~<K,V>~> iterator()
        +int size()
        +void clear()
    }

    HashMap "1" --> "*" Node : contains
    HashMap "1" --> "1" EntrySet : provides
    Node <|-- TreeNode : subclass (for Tree bins)

    class TreeNode~K,V~ {
        +TreeNode~K,V~ parent
        +TreeNode~K,V~ left
        +TreeNode~K,V~ right
        +boolean red
        +TreeNode~K,V~ getTreeNode(int hash, Object key)
    }

```

###  HashMap Internal Structure Overview

```mermaid
graph TD
  A[HashMap Object] --> B[table Array of Buckets]
  B --> C1[Bucket 0]
  B --> C2[Bucket 1]
  B --> C3[Bucket n]
  C1 --> D1[null or LinkedList/Tree of Nodes]
  C2 --> D2[null or LinkedList/Tree of Nodes]
  C3 --> D3[null or LinkedList/Tree of Nodes]
  D2 --> E1[Node<K,V> - key1, value1, hash, next]
  D2 --> E2[Node<K,V> - key2, value2, hash, next]
  E2 --> E3[Red-Black Tree Node<K,V> if treeified]
```

**Explanation:**

* `table[]`: The main array where each slot (bucket) can hold a linked list or a red-black tree.
* Each **bucket** is determined by the **hash of the key**.
* If too many collisions occur in a bucket (typically >8 elements), it **treeifies** into a **Red-Black Tree** for faster lookup.

---

## `put(K, V)` Operation 

```mermaid
sequenceDiagram
    participant C as Client Code
    participant H as HashMap
    participant A as table[] (Array)
    participant B as Bucket[i]
    participant N as Node (LinkedList or Tree)

    C->>H: put(key, value)
    H->>H: Compute hash = hash(key)
    H->>A: Find bucket index = (n - 1) & hash
    A->>B: Access bucket[i]

    alt Bucket is empty
        B->>H: Create new Node(key, value)
        H->>A: Insert Node into bucket[i]
    else Bucket not empty
        alt Bucket is LinkedList
            H->>B: Traverse linked list comparing keys
            alt Key exists
                B->>H: Replace existing value
            else New key
                B->>H: Append new Node to end
                H->>H: If size > TREEIFY_THRESHOLD → Treeify
            end
        else Bucket is Red-Black Tree
            H->>B: Insert into tree using compareTo()
            B->>H: Rebalance tree if needed
        end
    end

    H->>H: size++, check load factor
    alt Resize required
        H->>A: Rehash and expand table[]
    end
    H-->>C: Return old value (if replaced)
```

**Key components involved:**

* **Hash computation:** `hash = key.hashCode() ^ (hash >>> 16)`
* **Index calculation:** `(n - 1) & hash`
* **Node insertion:** Into linked list or tree
* **Resize:** If `size > capacity * loadFactor` (default 0.75)
* **Treeify:** When a bucket’s list exceeds threshold (8 nodes)

---

## `get(K)` Operation 

```mermaid
sequenceDiagram
    participant C as Client Code
    participant H as HashMap
    participant A as table[] (Array)
    participant B as Bucket[i]
    participant N as Node (LinkedList or Tree)

    C->>H: get(key)
    H->>H: Compute hash = hash(key)
    H->>A: index = (n - 1) & hash
    A->>B: Access bucket[i]

    alt Bucket is empty
        B-->>C: Return null
    else
        alt Bucket is LinkedList
            H->>B: Traverse nodes comparing key.equals()
            B-->>C: Return value if found
        else Bucket is Red-Black Tree
            H->>B: Perform tree search (compareTo)
            B-->>C: Return value if found
        end
    end
```

**Key points:**

* Only **bucket index** needs to be computed; no need to check entire table.
* Each lookup is O(1) average, O(log n) worst (if treeified).
* **No structural modification**, so no resizing.


##  Summary 

| Component          | Description                                 | Used in               |
| :----------------- | :------------------------------------------ | :-------------------- |
| **table[]**        | Main array storing buckets                  | Core structure        |
| **bucket**         | Slot in array holding a linked list or tree | Each hash index       |
| **Node<K,V>**      | Entry object storing key, value, hash, next | Linked list           |
| **Linked List**    | Used when few collisions                    | Default               |
| **Red-Black Tree** | Used when many collisions                   | After treeification   |
| **hash()**         | Mixes bits of key’s hashCode                | For even distribution |
| **resize()**       | Expands capacity when threshold exceeded    | After insertions      |


##  Visualization — HashMap Bucket Evolution

```mermaid
graph TD
  A[Bucket i] --> B1[Node1 ]
  B1 --> B2[Node2 ]
  B2 --> B3[Node3 ]
  B3 --> B4[Node4 ]
  B4 --> B5[...]
  B5 --> C[Treeify threshold reached ]
  C --> D[Red-Black Tree Nodes]
```

✅ **In summary:**

* `HashMap` combines **array + linked list + red-black tree** for optimal performance.
* `put()` adds or updates entries; may **treeify** or **resize**.
* `get()` traverses the corresponding **bucket** using key’s hash and equality.
* Average lookup/insertion time = **O(1)**; worst-case (treeified) = **O(log n)**.

---

Would you like me to extend this diagram to include **ConcurrentHashMap’s segment/stripe structure** — showing how it differs internally from HashMap during `put()` and `get()`?
