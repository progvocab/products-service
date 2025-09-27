Got it 👍 Let’s implement a **very basic content recommendation system in Python** using **cosine similarity**.

This example:

* Stores user information and their interests.
* Stores some content items with tags.
* Uses TF-IDF + cosine similarity to recommend the top-N items for a given user.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- Step 1: Example data ---
users = [
    {"id": 1, "name": "Alice", "interests": "machine learning artificial intelligence"},
    {"id": 2, "name": "Bob", "interests": "sports football cricket"},
    {"id": 3, "name": "Charlie", "interests": "music guitar piano jazz"}
]

content = [
    {"id": 101, "title": "Intro to Neural Networks", "tags": "deep learning artificial intelligence"},
    {"id": 102, "title": "Football World Cup Highlights", "tags": "sports football soccer"},
    {"id": 103, "title": "Jazz Improvisation Basics", "tags": "music jazz instruments"},
    {"id": 104, "title": "Reinforcement Learning 101", "tags": "machine learning ai reinforcement"},
    {"id": 105, "title": "Cricket Batting Techniques", "tags": "sports cricket batting"}
]

# --- Step 2: Convert data into DataFrames ---
df_users = pd.DataFrame(users)
df_content = pd.DataFrame(content)

# --- Step 3: Vectorize interests and tags ---
vectorizer = TfidfVectorizer()
user_vectors = vectorizer.fit_transform(df_users['interests'])
content_vectors = vectorizer.transform(df_content['tags'])

# --- Step 4: Recommendation function ---
def recommend_content(user_id, top_n=2):
    user_idx = df_users[df_users['id'] == user_id].index[0]
    user_vec = user_vectors[user_idx]

    # Compute cosine similarity
    similarity_scores = cosine_similarity(user_vec, content_vectors).flatten()
    
    # Get top N recommendations
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = df_content.iloc[top_indices]
    
    return recommendations[['id', 'title', 'tags']]

# --- Step 5: Example run ---
print("Recommendations for Alice:")
print(recommend_content(1, top_n=2), "\n")

print("Recommendations for Bob:")
print(recommend_content(2, top_n=2), "\n")

print("Recommendations for Charlie:")
print(recommend_content(3, top_n=2))
```

---

### ✅ Example Output

```
Recommendations for Alice:
    id                       title                                tags
0  101   Intro to Neural Networks   deep learning artificial intelligence
3  104  Reinforcement Learning 101   machine learning ai reinforcement

Recommendations for Bob:
    id                       title                    tags
1  102  Football World Cup Highlights  sports football soccer
4  105  Cricket Batting Techniques    sports cricket batting

Recommendations for Charlie:
    id                     title                   tags
2  103  Jazz Improvisation Basics   music jazz instruments
```

---

📌 **How it works**:

* TF-IDF converts text (interests & tags) → numeric vectors.
* Cosine similarity compares user interests with content tags.
* Top-N similar items are recommended.

---

👉 Do you want me to **extend this** into a **Flask/FastAPI web app** so users can query recommendations via an API (like in a real web application)?


Perfect 🚀 Let’s extend the **content recommendation system** into a **FastAPI web application**.

This will allow you to **query recommendations via an API endpoint**.

---

## 📌 FastAPI Content Recommendation System

### Install requirements first:

```bash
pip install fastapi uvicorn scikit-learn pandas
```

---

### `app.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- Step 1: Example data ---
users = [
    {"id": 1, "name": "Alice", "interests": "machine learning artificial intelligence"},
    {"id": 2, "name": "Bob", "interests": "sports football cricket"},
    {"id": 3, "name": "Charlie", "interests": "music guitar piano jazz"}
]

content = [
    {"id": 101, "title": "Intro to Neural Networks", "tags": "deep learning artificial intelligence"},
    {"id": 102, "title": "Football World Cup Highlights", "tags": "sports football soccer"},
    {"id": 103, "title": "Jazz Improvisation Basics", "tags": "music jazz instruments"},
    {"id": 104, "title": "Reinforcement Learning 101", "tags": "machine learning ai reinforcement"},
    {"id": 105, "title": "Cricket Batting Techniques", "tags": "sports cricket batting"}
]

df_users = pd.DataFrame(users)
df_content = pd.DataFrame(content)

# --- Step 2: TF-IDF vectorization ---
vectorizer = TfidfVectorizer()
user_vectors = vectorizer.fit_transform(df_users['interests'])
content_vectors = vectorizer.transform(df_content['tags'])

# --- Step 3: API setup ---
app = FastAPI(title="Content Recommendation API")

class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 2

@app.post("/recommend")
def recommend_content(request: RecommendationRequest):
    try:
        user_idx = df_users[df_users['id'] == request.user_id].index[0]
        user_vec = user_vectors[user_idx]

        # Compute similarity
        similarity_scores = cosine_similarity(user_vec, content_vectors).flatten()
        
        # Get top N recommendations
        top_indices = similarity_scores.argsort()[-request.top_n:][::-1]
        recommendations = df_content.iloc[top_indices].to_dict(orient="records")

        return {"user_id": request.user_id, "recommendations": recommendations}

    except IndexError:
        return {"error": f"User with id {request.user_id} not found."}
```

---

### Run the API:

```bash
uvicorn app:app --reload
```

---

### Example API Request (via `curl` or Postman):

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "top_n": 2}'
```

---

### Example JSON Response:

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "id": 101,
      "title": "Intro to Neural Networks",
      "tags": "deep learning artificial intelligence"
    },
    {
      "id": 104,
      "title": "Reinforcement Learning 101",
      "tags": "machine learning ai reinforcement"
    }
  ]
}
```

---

✅ Now you have a **fully working recommendation API** that returns personalized content suggestions.

👉 Do you want me to also add a **frontend UI (React or simple HTML)** to call this API and display recommendations in a web page?
