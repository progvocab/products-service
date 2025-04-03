## **Sparse Graph vs Dense Graph**
A **Sparse Graph** is a graph where the number of edges is much **less than the maximum possible edges**.

### **1️⃣ Sparse Graph Characteristics**
- If **E ≪ V²** → The graph is **sparse**.
- Example: A social network where each person (node) has only a few friends (edges).

### **2️⃣ Dense Graph Characteristics**
- If **E ≈ V²** → The graph is **dense**.
- Example: A **fully connected graph** (every node connects to every other node).

### **3️⃣ Adjacency Matrix for Graph Representation**
An **Adjacency Matrix** is a **V × V matrix** where:
- `matrix[i][j] = 1` (if an edge exists between vertex `i` and `j`).
- `matrix[i][j] = 0` (if no edge exists).

### **✅ Java Implementation - Adjacency Matrix**
```java
import java.util.Arrays;

class GraphMatrix {
    private int[][] adjMatrix;
    private int numVertices;

    // Constructor: Initialize matrix
    public GraphMatrix(int numVertices) {
        this.numVertices = numVertices;
        adjMatrix = new int[numVertices][numVertices];
    }

    // Add an edge
    public void addEdge(int src, int dest) {
        adjMatrix[src][dest] = 1;
        adjMatrix[dest][src] = 1; // Remove this line for directed graphs
    }

    // Remove an edge
    public void removeEdge(int src, int dest) {
        adjMatrix[src][dest] = 0;
        adjMatrix[dest][src] = 0;
    }

    // Print adjacency matrix
    public void printMatrix() {
        for (int[] row : adjMatrix) {
            System.out.println(Arrays.toString(row));
        }
    }

    public static void main(String[] args) {
        GraphMatrix graph = new GraphMatrix(5); // 5 vertices (0 to 4)
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);
        graph.addEdge(3, 4);

        graph.printMatrix();
    }
}
```

### **🔹 Output (Adjacency Matrix for 5 nodes)**
```
[0, 1, 1, 0, 0]
[1, 0, 0, 1, 0]
[1, 0, 0, 1, 0]
[0, 1, 1, 0, 1]
[0, 0, 0, 1, 0]
```

---

### **🔹 When to Use Adjacency Matrix vs Adjacency List?**
| Feature            | **Adjacency Matrix**  | **Adjacency List** |
|--------------------|----------------------|--------------------|
| **Best for**       | Dense Graphs (many edges) | Sparse Graphs (few edges) |
| **Space Complexity** | O(V²)                 | O(V + E) |
| **Edge Lookup**    | O(1) (direct indexing) | O(E/V) (search in list) |
| **Memory Usage**   | High (wastes space for sparse graphs) | Efficient for sparse graphs |

✅ **Use Adjacency Matrix for** **Dense Graphs** (complete graphs, game grids).  
✅ **Use Adjacency List for** **Sparse Graphs** (social networks, road networks).  

Would you like **weighted graph support**? 🚀