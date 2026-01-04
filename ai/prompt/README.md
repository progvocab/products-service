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

