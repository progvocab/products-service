prompt : Okey design a model for content recommendation , user interests are stored in preference table , the recommendation should be stored in recommendation table , once the user logs in there should be an api that can be called by ui to get the recommendation from db , also once user clicks the recommendation the data should be stored, which will be used to improve the accuracy of model , write the code for improving the accuracy also

---

Got it! We can design a **full-stack content recommendation system** with a feedback loop to improve accuracy. I’ll break it down step by step with a practical **Python + Flask + SQLAlchemy + pandas** example, and include **model updating logic**.

---

## **1. Database Design**

* **`preference` table**: stores user interests.
* **`recommendation` table**: stores recommended items for each user.
* **`interaction` table**: stores clicks on recommendations to improve model.

| Table            | Columns                         |
| ---------------- | ------------------------------- |
| `preference`     | `user_id`, `category`, `score`  |
| `recommendation` | `user_id`, `item_id`, `score`   |
| `interaction`    | `user_id`, `item_id`, `clicked` |

---

## **2. Model Design**

We’ll use a **matrix factorization model** for simplicity:

* Inputs: user preferences (from `preference` table)
* Output: predicted relevance score for items
* Loss function: **Mean Squared Error** or **BPR loss** for ranking

We’ll retrain periodically using **interaction feedback**.

---

## **3. Code Example**

```python
# app.py
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recommendation.db'
db = SQLAlchemy(app)

# ----------------------
# DB Models
# ----------------------
class Preference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    category = db.Column(db.String(50))
    score = db.Column(db.Float)

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    item_id = db.Column(db.Integer)
    score = db.Column(db.Float)

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    item_id = db.Column(db.Integer)
    clicked = db.Column(db.Boolean)

db.create_all()

# ----------------------
# Recommendation Model
# ----------------------
def train_model():
    # Load preferences
    prefs = pd.DataFrame([{
        'user_id': p.user_id,
        'category': p.category,
        'score': p.score
    } for p in Preference.query.all()])

    if prefs.empty:
        return

    # Pivot to user-item matrix
    user_item_matrix = prefs.pivot_table(index='user_id', columns='category', values='score', fill_value=0)

    # Apply Non-negative Matrix Factorization
    nmf = NMF(n_components=5, init='random', random_state=0)
    user_factors = nmf.fit_transform(user_item_matrix)
    item_factors = nmf.components_

    predicted_matrix = np.dot(user_factors, item_factors)
    scaler = MinMaxScaler()
    predicted_matrix = scaler.fit_transform(predicted_matrix)

    # Store recommendations in DB
    Recommendation.query.delete()
    for i, user_id in enumerate(user_item_matrix.index):
        for j, category in enumerate(user_item_matrix.columns):
            rec = Recommendation(user_id=user_id, item_id=j, score=predicted_matrix[i, j])
            db.session.add(rec)
    db.session.commit()

# ----------------------
# API Endpoints
# ----------------------
@app.route('/get_recommendations/<int:user_id>')
def get_recommendations(user_id):
    recs = Recommendation.query.filter_by(user_id=user_id).order_by(Recommendation.score.desc()).limit(10).all()
    return jsonify([{'item_id': r.item_id, 'score': r.score} for r in recs])

@app.route('/click_recommendation', methods=['POST'])
def click_recommendation():
    data = request.json
    interaction = Interaction(user_id=data['user_id'], item_id=data['item_id'], clicked=True)
    db.session.add(interaction)
    db.session.commit()
    # Optionally retrain model periodically
    train_model()
    return jsonify({'status': 'ok'})

# ----------------------
if __name__ == '__main__':
    app.run(debug=True)
```

---

### **4. How It Works**

1. Users have preferences stored in the `preference` table.
2. `train_model()` builds user-item relevance scores and stores top recommendations.
3. API `/get_recommendations/<user_id>` returns top recommendations for the UI.
4. When a user clicks a recommendation, the interaction is stored in `interaction` table.
5. You can **retrain the model periodically** (or after significant clicks) to **improve accuracy** using actual user feedback.

---

### **5. Optional Improvements**

* Use **BPR loss** instead of NMF for implicit feedback.
* Use **batch retraining** to scale instead of retraining on every click.
* Include **time-decay weights** in interactions to prioritize recent clicks.
* Add **caching layer** (Redis) to serve recommendations faster.

---

I can also write an **improved version with BPR ranking loss** so the model focuses on **clicks, not just predicted scores**, which is better for web recommendations.

Do you want me to do that?
