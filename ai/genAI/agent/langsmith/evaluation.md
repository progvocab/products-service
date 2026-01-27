Here’s a **clean, working Python example** showing how to do **LangSmith evaluation** for an LLM using **custom evaluators**. This is the most common, real-world setup.

---

## 1️⃣ Install & setup

```bash
pip install langsmith langchain openai
```

Set env vars:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=YOUR_LANGSMITH_API_KEY
export LANGCHAIN_PROJECT="llm-eval-demo"
export OPENAI_API_KEY=YOUR_OPENAI_KEY
```

---

## 2️⃣ Define the function (system under test)

This is what LangSmith will evaluate.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def answer_question(inputs: dict) -> dict:
    question = inputs["question"]
    response = llm.invoke(question)
    return {"answer": response.content}
```

---

## 3️⃣ Create a dataset in LangSmith

```python
from langsmith import Client

client = Client()

dataset = client.create_dataset(
    dataset_name="qa-eval-dataset",
    description="Simple QA evaluation dataset"
)

client.create_examples(
    inputs=[
        {"question": "What is Kafka used for?"},
        {"question": "Explain eventual consistency."}
    ],
    outputs=[
        {"answer": "Distributed event streaming platform"},
        {"answer": "Consistency model where replicas converge over time"}
    ],
    dataset_id=dataset.id
)
```

---

## 4️⃣ Define evaluators (core part)

### ✅ Exact match evaluator

```python
def exact_match(run, example):
    predicted = run.outputs["answer"].lower()
    expected = example.outputs["answer"].lower()

    return {
        "key": "exact_match",
        "score": int(expected in predicted)
    }
```

---

### ✅ LLM-based evaluator (recommended)

```python
from langchain_openai import ChatOpenAI

judge = ChatOpenAI(model="gpt-4o-mini")

def llm_judge(run, example):
    prompt = f"""
    Question: {example.inputs['question']}
    Expected Answer: {example.outputs['answer']}
    Model Answer: {run.outputs['answer']}

    Score the model answer from 0 to 1 for correctness.
    Return only the number.
    """

    score = float(judge.invoke(prompt).content.strip())

    return {
        "key": "llm_judge",
        "score": score
    }
```

---

## 5️⃣ Run evaluation

```python
from langsmith.evaluation import evaluate

evaluate(
    answer_question,
    data=dataset.name,
    evaluators=[exact_match, llm_judge],
)
```

---

## 6️⃣ View results

* Open **LangSmith UI**
* Go to **Projects → llm-eval-demo**
* You’ll see:

  * Traces
  * Scores per run
  * Aggregated metrics

---

## 🔥 What this gives you (architect-level insight)

* Offline, repeatable evaluation
* Mix of **rule-based + LLM-as-judge**
* Dataset versioning
* Regression testing for prompt/model changes

---

If you want next:

* **RAG evaluation (faithfulness, hallucination)**
* **Pairwise / A-B model comparison**
* **Human-in-the-loop scoring**
* **CI-based eval (fail build if score drops)**

tell me which one and I’ll give production-grade code.
