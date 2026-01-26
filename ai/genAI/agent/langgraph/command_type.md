**`Command` in LangGraph** is used to **explicitly control graph execution from inside a node**.

**Why it exists (core use):**

1. It lets a node **override normal edges** and choose the *next node* programmatically.
2. It allows **state updates + routing decisions** in one return value.
3. It’s useful for **dynamic workflows** where the next step isn’t known at graph design time.
4. It avoids complex conditional-edge wiring for highly dynamic paths.
5. It’s commonly used in **agent nodes, routers, retries, and tool-dispatch logic**.

**In one line:**

> `Command` gives a node **imperative control over “what happens next”**, instead of relying only on static edges.
