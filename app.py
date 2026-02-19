import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add project root to Python path (for Colab)
sys.path.append("/content/ai-teaching-assistant")

from models.difficulty_predictor import predict_difficulty
from learning_path.student_state import StudentState
from learning_path.recommendation_policy import recommend_learning_path


# -----------------------------
# Load Data
# -----------------------------

queries_df = pd.read_csv("ai-teaching-assistant/data/student_queries.csv")
logs_df = pd.read_csv("ai-teaching-assistant/data/student_learning_logs.csv")


# -----------------------------
# Create Embeddings
# -----------------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(queries_df["query_text"].tolist())


# -----------------------------
# Train Intent Classifier
# -----------------------------

y_intent = queries_df["intent"]

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X, y_intent, test_size=0.3, random_state=42
)

intent_model = LogisticRegression(max_iter=1000)
intent_model.fit(X_train_i, y_train_i)


# -----------------------------
# Train Topic Classifier
# -----------------------------

y_topic = queries_df["topic"]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X, y_topic, test_size=0.3, random_state=42
)

topic_model = LogisticRegression(max_iter=1000)
topic_model.fit(X_train_t, y_train_t)


# -----------------------------
# Main Function
# -----------------------------

def run_teaching_assistant(student_id, query):

    embedding = embedder.encode([query])

    intent = intent_model.predict(embedding)[0]
    topic = topic_model.predict(embedding)[0]
    difficulty = predict_difficulty(topic, intent)

    state = StudentState(student_id, logs_df).get_state()
    recommendation = recommend_learning_path(state)

    return {
        "query_understanding": {
            "intent": intent,
            "topic": topic,
            "difficulty_level": difficulty
        },
        "learning_recommendation": recommendation
    }


# -----------------------------
# Run Example
# -----------------------------

if __name__ == "__main__":

    test_queries = [
        "i dont understand how backpropagation works",
        "give me an example of gradient descent",
        "quick revision of linear regression",
        "i am confused about overfitting and regularization"
    ]

    for q in test_queries:
        print("\nQuery:", q)
        output = run_teaching_assistant(student_id=1, query=q)

        print("Intent:", output["query_understanding"]["intent"])
        print("Topic:", output["query_understanding"]["topic"])
        print("Difficulty:", output["query_understanding"]["difficulty_level"])

        rec = output["learning_recommendation"]
        print("Next Topic:", rec["next_topic"])
        print("Action:", rec["action"])
        print("Difficulty Adjustment:", rec["difficulty_adjustment"])

        print("-" * 50)



