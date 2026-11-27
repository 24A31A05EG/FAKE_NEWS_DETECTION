
import joblib
import pandas as pd

from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from src.features.tfidf_features import NewsTfidf


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load existing train and validation datasets
train = pd.read_csv(DATA_DIR / "isot_train.csv")
val = pd.read_csv(DATA_DIR / "isot_val.csv")

# Create two training examples per article:
# 1. Headline only
# 2. Full article
headline_texts = train["clean_title"].fillna("").astype(str)
article_texts = train["combined_text"].fillna("").astype(str)

mixed_texts = pd.concat(
    [headline_texts, article_texts],
    ignore_index=True
)

mixed_labels = pd.concat(
    [train["label"], train["label"]],
    ignore_index=True
)

# Train a separate TF-IDF vectorizer
vectorizer = NewsTfidf()
X_train = vectorizer.fit_transform(mixed_texts)

# Train a separate Linear SVM model
model = LinearSVC(class_weight="balanced", random_state=42)
model.fit(X_train, mixed_labels)

# Save new experiment artifacts separately
joblib.dump(model, MODEL_DIR / "linearsvm_mixed.joblib")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer_mixed.joblib")


def evaluate_input_type(name, texts):
    X = vectorizer.transform(texts)
    y_true = val["label"]
    predictions = model.predict(X)
    scores = model.decision_function(X)

    metrics = {
        "input_type": name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }

    return metrics


results = [
    evaluate_input_type(
        "Headline Only",
        val["clean_title"].fillna("").astype(str)
    ),
    evaluate_input_type(
        "Full Article",
        val["combined_text"].fillna("").astype(str)
    ),
]

results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORT_DIR / "mixed_input_validation.csv",
    index=False
)

print("\nMIXED INPUT VALIDATION RESULTS")
print(results_df.to_string(index=False))

print("\nNew model and vectorizer saved successfully.")
