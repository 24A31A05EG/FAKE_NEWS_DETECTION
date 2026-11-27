
import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

DATA_PATH = Path("data/processed/isot_test.csv")
MODEL_PATH = Path("models/linearsvm_mixed.joblib")
VECTORIZER_PATH = Path("models/tfidf_vectorizer_mixed.joblib")
OUTPUT_PATH = Path("reports/metrics/mixed_input_test_results.json")

# Load test data and saved model
test = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

results = {}

def evaluate(name, texts):
    X = vectorizer.transform(texts)
    y_true = test["label"]
    predictions = model.predict(X)
    scores = model.decision_function(X)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 6),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }

    results[name] = metrics

    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


evaluate(
    "headline_only",
    test["clean_title"].fillna("").astype(str)
)

evaluate(
    "full_article",
    test["combined_text"].fillna("").astype(str)
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nMixed-input test results saved successfully.")
