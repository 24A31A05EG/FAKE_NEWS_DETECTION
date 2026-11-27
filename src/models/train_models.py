
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.features.tfidf_features import NewsTfidf


DATA = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "isot_train.csv")
val = pd.read_csv(DATA / "isot_val.csv")

X_train_text = train["combined_text"].fillna("")
X_val_text = val["combined_text"].fillna("")
y_train = train["label"]
y_val = val["label"]

vectorizer = NewsTfidf()
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "NaiveBayes": MultinomialNB(),
    "LinearSVM": LinearSVC(class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
        class_weight="balanced"
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1_score": f1_score(y_val, predictions, zero_division=0)
    })

    joblib.dump(model, MODEL_DIR / f"{name.lower()}.joblib")
    print(classification_report(y_val, predictions, zero_division=0))

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
results_df.to_csv(REPORT_DIR / "ml_model_comparison.csv", index=False)

print("\nVALIDATION MODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nModels and metrics saved successfully.")
