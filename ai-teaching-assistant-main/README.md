# 🧠 AI-Powered Adaptive Teaching Assistant

An intelligent teaching assistant system that understands student queries and generates personalized learning recommendations using semantic embeddings and adaptive progression logic.

---

## 📌 Project Overview

This project implements a hybrid AI system that:

- Understands student queries using semantic embeddings (Sentence-BERT)
- Classifies query intent and topic using machine learning
- Predicts difficulty level dynamically
- Models student learning state using historical performance
- Generates personalized next-step recommendations

The system combines NLP-based query understanding with performance-aware adaptive learning.

---

## 🏗️ System Architecture

The system consists of two major components:

### 1️⃣ Query Understanding Module
- SentenceTransformer (all-MiniLM-L6-v2) for semantic embeddings
- Logistic Regression for:
  - Intent classification
  - Topic classification
- Rule-based difficulty adjustment

### 2️⃣ Adaptive Learning Path Module
- Student performance analysis from learning logs
- State modeling (quiz score, attempts, confidence)
- Rule-based recommendation engine for:
  - Revision
  - Advancement
  - Continuation

---

## 🔍 Key Highlights

- Semantic query understanding using Sentence-BERT embeddings
- Dual classifier system (Intent + Topic)
- Dynamic difficulty adjustment logic
- Student performance-aware progression system
- Modular and scalable architecture

---

## 📂 Project Structure

```
ai-teaching-assistant/
│
├── app.py
│
├── data/
│   ├── student_queries.csv
│   └── student_learning_logs.csv
│
├── models/
│   ├── __init__.py
│   ├── difficulty_predictor.py
│   ├── intent_classifier.py
│   ├── topic_classifier.py
│   └── embeddings.py
│
├── learning_path/
│   ├── __init__.py
│   ├── student_state.py
│   └── recommendation_policy.py
```



---

## ⚙️ How It Works

1. User submits a learning query.
2. Query is converted into embeddings using Sentence-BERT.
3. Intent and topic are predicted using trained classifiers.
4. Difficulty level is dynamically adjusted.
5. Student performance history is analyzed.
6. A personalized learning recommendation is generated.

---

## 🚀 How to Run

```bash
pip install sentence-transformers scikit-learn pandas
python ai-teaching-assistant/app.py

Query: i dont understand how backpropagation works
Intent: Explanation
Topic: Backpropagation
Difficulty: Intermediate
Next Topic: Backpropagation
Action: Revision
Difficulty Adjustment: Decrease


---





