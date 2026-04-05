

**Interpretability vs Explainability — First, the Difference:**

| | Interpretability | Explainability |
|---|---|---|
| **Definition** | How well humans can **understand** the model's internal mechanics | How well you can **explain** why a model made a specific prediction |
| **Focus** | Model structure | Model output |
| **Question** | "How does this model work?" | "Why did this model predict X?" |
| **Example** | Decision tree is interpretable by design | SHAP explains why BERT classified this email as spam |

---

**Category 1 — Intrinsically Interpretable Models:**
These models are interpretable **by design** — no extra tools needed:

| Model | Why Interpretable | Limitation |
|---|---|---|
| **Linear Regression** | Each feature has a coefficient showing exact contribution | Assumes linear relationships |
| **Logistic Regression** | Coefficients show feature importance and direction | Limited to linear boundaries |
| **Decision Tree** | Visual tree structure shows exact decision path | Gets complex when deep |
| **Decision Rules** | IF-THEN rules humans can read directly | Doesn't scale well |
| **Naive Bayes** | Probability calculations are fully transparent | Strong independence assumption |

**Example — Linear Regression:**
```
House Price = 50,000
           + (200 × sqft)
           + (10,000 × bedrooms)
           - (5,000 × crime_rate)

→ Instantly clear which features matter and how much
```

---

**Category 2 — Black Box Models (Need Explainability Tools):**
These models are powerful but not naturally interpretable:

| Model | Why Black Box |
|---|---|
| **Neural Networks / Deep Learning** | Millions of weights — impossible to interpret directly |
| **Random Forest** | Hundreds of trees combined |
| **XGBoost / Gradient Boosting** | Complex ensemble of trees |
| **SVM** | High dimensional transformations |
| **BERT / LLMs** | Billions of parameters |

---

**Category 3 — Explainability Techniques:**

---

**3A. Global Explainability**
Explains model behavior **overall** — across all predictions:

| Technique | How it works | Use case |
|---|---|---|
| **Feature Importance** | Ranks which features matter most overall | "sqft is most important for house prices" |
| **Partial Dependence Plots (PDP)** | Shows how one feature affects predictions on average | "How does age affect loan approval overall?" |
| **Permutation Importance** | Shuffles each feature and measures accuracy drop | "Which feature hurts most when removed?" |

---

**3B. Local Explainability**
Explains **individual predictions** — why did the model predict X for THIS specific instance:

| Technique | How it works | Use case |
|---|---|---|
| **SHAP** | Assigns contribution score to each feature for each prediction | "This loan was rejected because income was low (-0.4) and debt was high (-0.3)" |
| **LIME** | Builds simple local model around one prediction | "For this specific patient, age and BMI drove the prediction" |
| **Counterfactual** | "What would need to change to get a different prediction?" | "If income was $5,000 higher, loan would be approved" |
| **Attention Maps** | Highlights which words/pixels model focused on | "Model flagged this email as spam because of 'free money'" |

---

**SHAP — Most Important for Exam:**

SHAP (SHapley Additive exPlanations) is the gold standard for explainability:

```
Prediction: Loan REJECTED ❌

SHAP Values:
Income:        -0.42  (low income hurts)
Debt ratio:    -0.31  (high debt hurts)
Credit score:  +0.28  (good credit helps)
Employment:    +0.15  (stable job helps)
Age:           -0.08  (slightly hurts)
─────────────────────
Base value:    +0.50
Final score:    0.12  → REJECTED (below 0.5 threshold)
```

Every feature gets a **positive or negative contribution** to the final prediction — fully transparent! ✅

---

**Category 4 — AWS Tools for Explainability:**

---

**SageMaker Clarify — Most Important:**

| Feature | Detail |
|---|---|
| **Bias detection** | Detects bias in data and model predictions |
| **SHAP explanations** | Provides feature attribution for each prediction |
| **Pre-training bias** | Detects bias before model is trained |
| **Post-training bias** | Detects bias after model is trained |
| **Model cards** | Documents model behavior and limitations |
| **Works with** | Tabular, NLP, and computer vision models |

**SageMaker Clarify bias metrics:**

| Metric | What it measures |
|---|---|
| **Class Imbalance (CI)** | Is one group over/under-represented in data? |
| **Disparate Impact (DI)** | Do different groups get different prediction rates? |
| **DPPL** | Difference in positive prediction rates between groups |

---

**Amazon Bedrock Explainability:**

| Feature | Detail |
|---|---|| **Model evaluation** | Compare models on accuracy, robustness, toxicity |
| **Guardrails** | Control and explain model behavior |
| **Prompt transparency** | See exactly what prompt drove what output |

---

**Category 5 — Explainability by Model Type:**

| Model | Best Explainability Technique |
|---|---|
| **Linear/Logistic Regression** | Coefficients (built-in) |
| **Decision Tree** | Tree visualization (built-in) |
| **Random Forest** | Feature importance + SHAP |
| **XGBoost** | Feature importance + SHAP |
| **Neural Network** | SHAP + LIME + Attention maps |
| **LLMs** | Attention maps + prompt analysis |
| **Computer Vision** | Grad-CAM + saliency maps |

---

**Grad-CAM — For Computer Vision:**

```
Input Image: X-ray scan
        ↓
CNN Model predicts: Pneumonia (92%)
        ↓
Grad-CAM highlights:
[Shows heatmap over lung area
that drove the prediction]
        ↓
Doctor can verify model looked
at the RIGHT part of the image ✅
```

---

**Why Explainability Matters — Exam Context:**

| Domain | Why needed |
|---|---|
| **Finance** | Regulations require explaining loan rejections (ECOA, GDPR) |
| **Healthcare** | Doctors need to understand AI diagnosis reasoning |
| **Legal** | Court decisions need human-understandable justification |
| **HR** | Hiring algorithms must not discriminate |
| **Responsible AI** | AWS pillar — models must be transparent and fair |

---

**Key Exam Decision Framework:**

| Scenario | Answer |
|---|---|
| Need to explain individual prediction | **SHAP / LIME** |
| Need to explain overall model behavior | **Feature Importance / PDP** |
| Need to detect bias in data/model | **SageMaker Clarify** |
| Need interpretable model by design | **Linear Regression / Decision Tree** |
| Need to explain image classification | **Grad-CAM / Saliency Maps** |
| Need to explain LLM output | **Attention Maps / Prompt Analysis** |
| Regulatory compliance for AI decisions | **SageMaker Clarify + Model Cards** |

---

**Simple Memory Trick:**

```
Interpretability = Glass Box 🔍
(you can see inside the model)

Explainability = Translator 🗣️
(someone explains what black box did)

SHAP = Receipt 🧾
(itemized breakdown of every prediction)

Clarify = Auditor 🕵️
(checks for bias and fairness)
```

