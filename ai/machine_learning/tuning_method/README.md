### List of LLM tuning methods

* **Prompt Engineering** – Manually crafting instructions to guide model behavior without training.
* **In-Context Learning (Few-shot)** – Providing examples inside the prompt to steer responses.
* **Hard Prompt Tuning** – Using fixed, reusable instruction templates.
* **Soft Prompt Tuning** – Learning virtual prompt embeddings while freezing the base model.
* **Prefix Tuning** – Learning trainable prefix vectors injected into attention layers.
* **Adapter Tuning** – Training small adapter layers inserted into the model.
* **LoRA (Low-Rank Adaptation)** – Training low-rank matrices to adapt attention weights efficiently.
* **QLoRA** – LoRA combined with quantization for very low-memory training.
* **PEFT (Parameter-Efficient Fine-Tuning)** – Umbrella approach covering LoRA, adapters, prefixes, etc.
* **Full Fine-Tuning** – Updating all model weights with task-specific data.
* **Instruction Tuning** – Fine-tuning on instruction–response datasets for better task following.
* **RLHF** – Reinforcement learning using human feedback to align outputs.

**Rule of thumb:** start from the top; move down only if needed.


Below is a **simple, real-world Python example** showing how to **tune an existing LLM using LoRA (PEFT)**.
This is the **most common and recommended tuning approach today**.

---

## Example: Fine-tuning an existing Hugging Face model using **LoRA**

### 1️⃣ Install dependencies

```bash
pip install transformers datasets peft accelerate torch
```

---

### 2️⃣ Load base model and tokenizer (pretrained model)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

### 3️⃣ Configure LoRA (parameter-efficient tuning)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                     # rank (low-rank matrices)
    lora_alpha=32,
    target_modules=["c_attn"],  # attention layers in GPT-2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

👉 Only **~1–2% parameters** become trainable.

---

### 4️⃣ Prepare training data

```python
from datasets import Dataset

data = [
    {"text": "Store sale report for last week shows increased footfall."},
    {"text": "Customer bought phone and display accessories."}
]

dataset = Dataset.from_list(data)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)
```

---

### 5️⃣ Train the model

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
```

---

### 6️⃣ Inference with tuned model

```python
prompt = "Generate a store analytics summary:"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 🔍 What was tuned here (conceptually)

* Base LLM weights ❌ frozen
* LoRA low-rank matrices ✅ trained
* Fast, cheap, and production-friendly

---

## 🧠 When to use this

* Domain adaptation (store analytics, customer data)
* Limited training data
* No GPU clusters available

If you want, I can also show:

* **Adapter tuning**
* **Soft prompt tuning**
* **QLoRA (low-memory)**
* **Full fine-tuning comparison**

Just tell me 👍
