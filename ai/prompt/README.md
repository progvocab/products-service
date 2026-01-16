## *Example-based prompting*

### 1. Zero-shot prompting

* No examples provided
* Model relies on prior knowledge
* Fast to use
* Lower control
* Common for classification

### 2. Few-shot prompting

* Small number of examples given
* Improves accuracy
* Guides output format
* More prompt effort
* Limited scalability

### 3. One-shot prompting

* Exactly one example provided
* Faster than few-shot
* Better than zero-shot
* Moderate control
* Common in demos

### 4. Negative prompting

* Specifies what **not** to do
* Reduces unwanted outputs
* Improves precision
* Often combined with others
* Used in image/text generation

### 5. Instruction prompting

* Clear task instructions only
* Structured outputs
* Easy to maintain
* Works well with LLMs
* Business-friendly





## **Reason based prompts**

* **Chain of Thought (CoT):** Ask model to reason step-by-step.
* **Tree of Thought (ToT):** Explore multiple reasoning paths before choosing.
* **Self-Consistency:** Sample multiple CoTs and pick the best.
* **Role Prompting:** Assign a persona (e.g., “act as an auditor”).
* **Contextual Prompting:** Provide background/domain context.
* **Multi-modal Prompting:** Combine text + image/audio inputs.





* **Chain of Thought (CoT)** and **Tree of Thought (ToT)** focus on *how the model reasons*, not just how examples are given.
* Zero-shot / few-shot describe **example provisioning**, while CoT/ToT describe **reasoning control**.


## Components of Prompt 

* system-level rules
* task-level instructions.
* **Context / Background** – domain knowledge, business scenario, or constraints
* **Input Data** – text, images, audio, video provided for analysis
* **Examples (Shots)** – zero-shot, one-shot, or few-shot demonstrations
* **Output Schema / Format** – JSON, table, bullets, or strict templates
* **Reasoning Guidance** – CoT hints, step-by-step or verification instructions
* **Negative Constraints** – what to avoid or exclude
* **Tools / Grounding References** – retrieved documents, APIs, or databases
* **Role / Persona** – perspective the model should adopt
* **Evaluation Signals** – confidence thresholds, scoring rules, or labels


### Trade off

**Hard Prompt** and **Soft Prompt** are two ways to guide large language models, with different trade-offs.

**Hard Prompt**

* Uses **fixed, human-written text instructions** (natural language).
* Examples: “Classify this image as phone or box”, “Summarize the text”.
* Easy to create, no training required.
* Changes require **manual rewriting**.

**Soft Prompt**

* Uses **learned embeddings (virtual tokens)** instead of words.
* Trained via backpropagation while keeping the base model frozen.
* More **parameter-efficient** and task-specific.
* Not human-readable but often more accurate.

**In short:** hard prompts are *written instructions*; soft prompts are *learned instructions*.


### GPT Model

The **GPT (Generative Pre-trained Transformer) model was created by OpenAI**.

More specifically:

* The **Transformer architecture** was introduced by **Google researchers** in the 2017 paper *“Attention Is All You Need”*.
* **OpenAI** adapted and extended this architecture to create **GPT**, first released as **GPT-1 in 2018**, followed by GPT-2, GPT-3, GPT-4, and later versions.

**One-line answer:**

> *GPT was created by OpenAI, based on the Transformer architecture originally proposed by Google researchers.*



Here are the **main parts of a GPT (Transformer) model**, each explained in **one line**:

1. **Tokenizer** – Converts raw text into discrete tokens the model can process.
2. **Token Embeddings** – Maps each token to a high-dimensional vector representing its meaning.
3. **Positional Embeddings** – Encodes the position of each token to preserve word order.
4. **Transformer Blocks** – Stacked layers that iteratively refine token representations.
5. **Self-Attention Mechanism** – Determines which tokens are most relevant to each other for context.
6. **Feed-Forward Networks (MLP)** – Applies non-linear transformations to enrich representations.
7. **Layer Normalization** – Stabilizes training by normalizing activations in each layer.
8. **Residual Connections** – Helps gradients flow and preserves information across layers.
9. **Output (LM) Head** – Projects hidden states to vocabulary logits for next-token prediction.
10. **Softmax Layer** – Converts logits into probabilities over the vocabulary.

**One-line takeaway:**

> *GPT is a tokenizer-driven, attention-based neural network that predicts the next token using stacked transformer layers.*

