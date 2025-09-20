 let’s build a **minimal ML Model Service** that computes **fraud probability** using **transaction, user, and device data**.

We’ll use:

* **scikit-learn** → for ML (Logistic Regression, lightweight and interpretable).
* **Flask** → to serve the model as a REST/gRPC-like API.
* **joblib** → for model persistence.

This will be a **tiny working example**, easy to extend later.

---

# 🔹 Step 1: Train and Save Fraud Detection Model

```python
# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

# --- Sample training data (transaction + user + device features) ---
# In real-world, this comes from transaction history
data = pd.DataFrame({
    "amount": [20, 5000, 50, 7000, 100, 6000],  # transaction amount
    "is_foreign": [0, 1, 0, 1, 0, 1],          # foreign transaction flag
    "device_trust": [1, 0, 1, 0, 1, 0],        # trusted device flag
    "prev_fraud": [0, 1, 0, 1, 0, 1],          # user fraud history
    "label": [0, 1, 0, 1, 0, 1]                # 1 = fraud, 0 = legit
})

X = data.drop("label", axis=1)
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model trained and saved!")
```

---

# 🔹 Step 2: Serve Fraud Detection Model

```python
# fraud_service.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load("fraud_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Input JSON example:
    {
      "amount": 5000,
      "is_foreign": 1,
      "device_trust": 0,
      "prev_fraud": 1
    }
    """
    data = request.json
    features = np.array([
        data["amount"],
        data["is_foreign"],
        data["device_trust"],
        data["prev_fraud"]
    ]).reshape(1, -1)

    # Predict fraud probability
    fraud_prob = model.predict_proba(features)[0][1]
    result = {
        "fraud_probability": float(fraud_prob),
        "is_fraud": int(fraud_prob > 0.5)  # threshold = 0.5
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
```

---

# 🔹 Step 3: Example Request

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 5000, "is_foreign": 1, "device_trust": 0, "prev_fraud": 1}'
```

### ✅ Example Response

```json
{
  "fraud_probability": 0.91,
  "is_fraud": 1
}
```

---

⚡ This service is small but **production-ready foundation**:

* Trains a fraud model.
* Saves and loads it with joblib.
* Exposes a prediction endpoint.
* Can be **containerized (Docker)** and integrated with payment gateway microservices.

---

Would you like me to **extend this service to use gRPC instead of REST (Flask)** so it matches the **real-time fraud detection architecture** we designed earlier?
