
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)

from src.features.tfidf_features import NewsTfidf

DATA = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/metrics")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

test = pd.read_csv(DATA / "isot_test.csv")
X_text = test["combined_text"].fillna("")
y_test = test["label"]

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
X_test = vectorizer.transform(X_text)

model_files = {
    "LogisticRegression": "logisticregression.joblib",
    "NaiveBayes": "naivebayes.joblib",
    "LinearSVM": "linearsvm.joblib",
    "RandomForest": "randomforest.joblib"
}

results = []

for name, filename in model_files.items():
    model = joblib.load(MODEL_DIR / filename)
    predictions = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
    }

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, scores)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        metrics["roc_auc"] = roc_auc_score(y_test, scores)

    results.append(metrics)
    print(f"\n{name}")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

with open(REPORT_DIR / "final_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nFinal test evaluation saved successfully.")
