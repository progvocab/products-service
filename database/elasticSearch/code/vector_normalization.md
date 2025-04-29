**Vector normalization** is the process of scaling a vector so that it has a **specific length (usually 1)** while maintaining its **direction**. This is commonly used in machine learning, natural language processing, and information retrieval—especially in **vector similarity** computations (like cosine similarity).

---

### **Purpose of Vector Normalization**

Normalization ensures:
- All vectors have the **same scale**.
- Comparison metrics (e.g., cosine similarity) aren't skewed by vector magnitude.
- Improved numerical stability and performance in models.

---

### **Types of Vector Normalization**

1. **L2 Normalization (Euclidean norm)**  
   Most common. Scales vector so that:
   \[
   \|v\|_2 = \sqrt{v_1^2 + v_2^2 + ... + v_n^2} = 1
   \]

   **Formula**:
   \[
   v_{\text{normalized}} = \frac{v}{\|v\|_2}
   \]

   **Example (Python)**:
   ```python
   import numpy as np

   v = np.array([3, 4])
   normalized_v = v / np.linalg.norm(v)
   print(normalized_v)  # Output: [0.6, 0.8]
   ```

2. **L1 Normalization (Manhattan norm)**  
   Sum of absolute values equals 1.

   \[
   \|v\|_1 = |v_1| + |v_2| + ... + |v_n|
   \]

3. **Max Norm (Infinity norm)**  
   Scales vector by its largest component.

---

### **Use Cases**

- **Cosine Similarity**: Cosine between normalized vectors is just their dot product.
- **Search Engines**: Elasticsearch, vector databases normalize vectors before similarity comparison.
- **Machine Learning**: To ensure features are on comparable scales.

---

Let me know if you'd like to see how normalization affects similarity scores or a full example using vector databases like FAISS or Elasticsearch.