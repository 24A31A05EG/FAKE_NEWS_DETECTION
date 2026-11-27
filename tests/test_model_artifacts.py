
from pathlib import Path
import joblib
import pandas as pd


def test_all_trained_model_files_exist():
    models = [
        "logisticregression.joblib",
        "naivebayes.joblib",
        "linearsvm.joblib",
        "randomforest.joblib",
        "tfidf_vectorizer.joblib",
    ]

    for model in models:
        assert (Path("models") / model).is_file()


def test_trained_models_can_be_loaded():
    models = [
        "logisticregression",
        "naivebayes",
        "linearsvm",
        "randomforest",
    ]

    for name in models:
        model = joblib.load(Path("models") / f"{name}.joblib")
        assert hasattr(model, "predict")


def test_vectorizer_can_transform_news():
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    sample = ["Government announces new economic policy today"]
    features = vectorizer.transform(sample)

    assert features.shape[0] == 1
    assert features.shape[1] > 0


def test_final_evaluation_report_exists():
    assert Path("reports/metrics/final_test_results.json").is_file()


def test_model_comparison_report_exists():
    report = Path("reports/metrics/ml_model_comparison.csv")
    assert report.is_file()

    results = pd.read_csv(report)
    assert len(results) == 4
    assert {"model", "accuracy", "precision", "recall", "f1_score"}.issubset(
        results.columns
    )
