
**Attention** in a Transformer is the mechanism that lets the model decide **which parts of the input to focus on** when processing each element.

* Each input is converted into **Query (Q), Key (K), and Value (V)** vectors.
* The model compares **queries with keys** to compute attention scores.
* These scores decide **how much importance** to give to each value.
* The weighted values are summed to form the output representation.
* This allows the model to capture **long-range relationships** in data.

**In simple terms:** attention lets the model *look at everything, decide what matters most, and use that information to understand context*.


**Multi-Head Attention** lets a Transformer look at the same input **in multiple ways at the same time**.

* Instead of one attention calculation, the input is split into **multiple heads**.
* Each head learns to focus on **different relationships** (e.g., position, meaning, structure).
* Every head has its own **Q, K, V projections**.
* The outputs of all heads are **concatenated and combined**.
* This gives a richer and more expressive representation than single attention.

**In simple terms:** multi-head attention allows the model to **pay attention to different aspects of the input simultaneously**.
