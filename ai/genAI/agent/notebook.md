Perfect 👌 — let’s turn the **local agent** into a **Jupyter Notebook** so you can run and experiment with it directly.

Here’s a notebook script you can save as `local_agent.ipynb` and open in Jupyter/Lab/VSCode.

---

## 📒 `local_agent.ipynb`

```python
# 🧑‍💻 Local LLM Agent with Ollama
# =================================
# This notebook demonstrates how to build and query a simple local agent
# using Ollama + Python (OpenAI-compatible API).

# ✅ Requirements:
# 1. Install Ollama: https://ollama.ai
# 2. Pull a model:   ollama pull llama3
# 3. pip install openai
```

```python
# Step 1: Import dependencies
from openai import OpenAI
```

```python
# Step 2: Connect to local Ollama server
# Ollama runs a local API at http://localhost:11434
client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")
```

```python
# Step 3: Define the Agent class
class LocalAgent:
    def __init__(self, model="llama3", temperature=0.7, top_p=0.9, max_tokens=512):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # System prompt defines the agent’s "role"
        self.system_prompt = (
            "You are a helpful AI assistant that explains answers clearly and concisely."
        )

    def query(self, user_input: str) -> str:
        """Send query to the local LLM via Ollama"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
```

```python
# Step 4: Create an instance of the agent
agent = LocalAgent(model="llama3", temperature=0.5, top_p=0.8, max_tokens=300)
```

```python
# Step 5: Run a test query
response = agent.query("Explain quantum computing in simple terms.")
print(response)
```

```python
# Step 6: Interactive loop inside notebook
# (Run this cell multiple times to chat interactively)
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    answer = agent.query(user_input)
    print("Agent:", answer)
```

---

## 🚀 Running the Notebook

1. Start Ollama in the background:

   ```bash
   ollama run llama3
   ```

   (or any other pulled model, e.g., `mistral`, `phi3`).
2. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
3. Open `local_agent.ipynb`, run cells, and chat with your **local agent** 🎉

---

Do you want me to also add a **memory module** (SQLite or in-memory history) to this notebook so the agent remembers past conversation turns?
