* Constrain prompts with **clear instructions and expected formats**.
* Use **grounding** by providing facts, data, or references in the prompt.
* Ask the model to **say “unknown” when information is missing**.
* Apply **verification steps** (self-check, consistency checks).
* Combine prompts with **external tools or databases (RAG)** for factual accuracy.

**Sample system prompt:**

> *You are an assistant that must answer using only the information explicitly provided by the user or attached documents. If the answer is not present or cannot be confidently derived, respond with “I don’t know based on the given information.” Do not guess, assume, or invent facts. Clearly state uncertainty when data is missing.*

### Grounding 

* **Grounding** means anchoring the model’s responses to **trusted, real data** instead of relying only on its memory.
* It reduces hallucinations by limiting the model to **provided facts or sources**.
* Grounding can use **documents, databases, or APIs** as reference.
* The model answers **based only on this supplied context**.
* Common examples include **RAG systems** and tool-based lookups.


