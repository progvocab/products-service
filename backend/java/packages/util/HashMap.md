

# HashMap
```mermaid
graph TD
    A["HashMap Table (Array of Bins)"] --> B0["Bin 0 (empty)"]
    A --> B1[Bin 1 → Bucket]
    A --> B2["Bin 2 (empty)"]
    A --> B3[Bin 3 → Bucket]
    A --> B4["Bin 4 (empty)"]

    B1 --> C1["(Node) K1:V1"]
    C1 --> C2["(Node) K9:V9"]

    B3 --> D1["(Node) K4:V4"]
    D1 --> D2["(Node) K20:V20"]
    D2 --> D3["(Node) K36:V36"]

    subgraph Treeified Bucket
        D1 -. treeify when size > 8 .-> T1[Red-Black Tree Nodes]
    end

    style B1 fill:#e0f7fa,stroke:#26a69a,stroke-width:2px
    style B3 fill:#e0f7fa,stroke:#26a69a,stroke-width:2px
    style T1 fill:#fce4ec,stroke:#ad1457,stroke-width:2px
```

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


 

A **resize (rehash)** occurs when:

> `size > capacity × loadFactor`

Where:

* **`size`** → Number of key-value pairs currently stored
* **`capacity`** → Current number of **buckets** (slots in the internal array)
* **`loadFactor`** → Threshold that controls when to resize (default **0.75**)

When this condition is met, the **HashMap doubles its capacity** and **rehashes** all existing entries.



##  **Default Values (in Java’s HashMap)**

| Property             | Default Value      |
| -------------------- | ------------------ |
| **Size**             | 0                 |
| **Initial Capacity** | 16                 |
| **Load Factor**      | 0.75f              |
| **Resize Threshold** | 16 × 0.75 = **12** |

So:

* When the 13th key-value pair is inserted → **resize occurs**.

 

##  **How It Changes**

Let’s trace it step-by-step:

| Step                 | Capacity | Load Factor | Threshold (Capacity × LoadFactor) | Size After Insert | Resize?                  |
| -------------------- | -------- | ----------- | --------------------------------- | ----------------- | ------------------------ |
| Start                | 16       | 0.75        | 12                                | 0                 | No                       |
| Insert 1 → 12        | 16       | 0.75        | 12                                | 1 → 12            | No                       |
| Insert 13th Entry    | 16       | 0.75        | 12                                | 13                |   Yes (Resize Triggered) |
| After Resize         | **32**   | 0.75        | **24**                            | 13                | No                       |
| Insert More Up to 24 | 32       | 0.75        | 24                                | 14 → 24           | No                       |
| Insert 25th Entry    | 32       | 0.75        | 24                                | 25                |  Yes (Resize Triggered) |
| After Resize         | **64**   | 0.75        | **48**                            | 25                | No                       |

##   **What Happens During Resize**

1. **New Capacity** = `oldCapacity * 2`
2. **New Threshold** = `newCapacity * loadFactor`
3. A **new bucket array** is created.
4. Each entry from the old array is **rehashed** into the new array (based on new indices).
5. This operation is **O(n)** and happens occasionally.

 
 



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
#### Put operation 
> When a new key pair is added hashmap takes the hashCode of the key and pass it to hash function with gives the index of this entey in bins arrays,  if there is a bucket already present , that means there Is a hash collision , now hashmap check the type of the bucket,  is it single object,  linked list or red black tree , if it's a single object , hashmap uses equals to check if both keys are same , if yes the value is overwritten,  else single object is converted to linked list and new entry is added at the end , if it is already a linked list then hashmap iterates over the linked list and find the matching key , if key is found value is updated , else new entry added at the end , if the threshold is crossed , meaning linked list has more than 8 elements,  it's converted to red black tree , new element is added and tree is balanced , if it's already a red black tree , hashmap traverses the red black tree , finds the key and overwrite the value , if not found hashmap adds the element to the red black tree and Balances it

Would you like me to extend this diagram to include **ConcurrentHashMap’s segment/stripe structure** — showing how it differs internally from HashMap during `put()` and `get()`?
