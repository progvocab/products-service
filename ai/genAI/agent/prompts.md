Got it 👍 — let’s go over the different types of **prompts/messages** that are used in systems like ChatGPT or other LLM-based applications. These are usually structured messages that guide how the model responds.

---

# 🧩 Types of Prompts / Messages

Most modern LLMs (like GPT-4/5) use a **chat-based prompt format**, where multiple roles exist:

---

## 1. **System Prompt**

* **Who sets it?** → Developer or platform (not the end-user).
* **Purpose** → Defines the **behavior, personality, and constraints** of the model.
* **Characteristics**:

  * High priority → The model must follow system instructions over user requests.
  * Used for alignment (e.g., "You are a helpful assistant").

📌 **Example:**

```json
{
  "role": "system",
  "content": "You are ChatGPT, a helpful AI assistant. Be concise, polite, and explain concepts with examples."
}
```

---

## 2. **User Prompt**

* **Who sets it?** → The end-user (like you right now).
* **Purpose** → Contains the actual **question, request, or instruction**.
* **Characteristics**:

  * Directly drives the model’s response.
  * Can be free-form or structured.

📌 **Example:**

```json
{
  "role": "user",
  "content": "Explain tensor factorization with an example."
}
```

---

## 3. **Assistant Message**

* **Who sets it?** → The model itself (previous responses).
* **Purpose** → Provides **conversation history**, so the model maintains context.
* **Characteristics**:

  * Not a new instruction, but helps maintain continuity.

📌 **Example:**

```json
{
  "role": "assistant",
  "content": "Tensor factorization generalizes matrix factorization by adding more dimensions..."
}
```

---

## 4. **Developer Prompt** (sometimes called **tool/system extension**)

* **Who sets it?** → Application developers.
* **Purpose** → Gives **secondary constraints or rules**, often hidden from users.
* **Characteristics**:

  * Lower priority than system, but higher than user.
  * Often used to enforce safety or formatting rules.

📌 **Example:**

```json
{
  "role": "developer",
  "content": "Always return answers in JSON format."
}
```

---

## 5. **Tool/Function Call Messages**

* **Who sets it?** → The model or developer.
* **Purpose** → When the model needs to call an external function, API, or tool.
* **Characteristics**:

  * Structured (not free text).
  * Enables integration with external systems (e.g., database query, web search).

📌 **Example:**

```json
{
  "role": "assistant",
  "tool_call": {
    "name": "getWeather",
    "arguments": {"location": "New York"}
  }
}
```

---

## 6. **Other Variants**

Depending on the platform, you might see additional roles:

* **System Context Extensions** → hidden safety/behavior instructions.
* **Critic / Reviewer Messages** → for self-reflection (used in some advanced LLMs).
* **Observation Messages** → results returned from tool calls.

---

# 🔑 Priority Order

Typically, the model follows:

**System > Developer > User > Assistant (history)**

So, if there’s a conflict (e.g., system says “be concise” but user says “write 1000 words”), the **system prompt wins**.

---

👉 This structure allows building complex apps (like your **recommender system UI**) where:

* System defines model’s behavior.
* User sends queries.
* Assistant answers.
* Developer enforces formatting.
* Tool calls fetch data (e.g., recommendations from DB).

---

Would you like me to draw a **Mermaid diagram showing how system, user, assistant, and tool prompts flow inside an LLM pipeline**?
