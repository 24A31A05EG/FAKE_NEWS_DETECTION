
import joblib
from pathlib import Path


def load_model_and_vectorizer():
    model = joblib.load(Path("models/linearsvm.joblib"))
    vectorizer = joblib.load(Path("models/tfidf_vectorizer.joblib"))
    return model, vectorizer


def test_model_predicts_single_news():
    model, vectorizer = load_model_and_vectorizer()
    text = ["Government announces new economic policy today"]
    features = vectorizer.transform(text)

    prediction = model.predict(features)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_model_predicts_multiple_news():
    model, vectorizer = load_model_and_vectorizer()
    texts = [
        "Government announces new economic policy today",
        "Scientists discover a new species in the ocean",
    ]

    features = vectorizer.transform(texts)
    predictions = model.predict(features)

    assert len(predictions) == 2


def test_prediction_is_numeric_label():
    model, vectorizer = load_model_and_vectorizer()
    features = vectorizer.transform(["A new scientific research report was published"])

    prediction = model.predict(features)[0]

    assert prediction in (0, 1)


def test_model_decision_function_returns_score():
    model, vectorizer = load_model_and_vectorizer()
    features = vectorizer.transform(["Central bank announces interest rate decision"])

    scores = model.decision_function(features)

    assert len(scores) == 1


def test_model_handles_headline_input():
    model, vectorizer = load_model_and_vectorizer()
    headline = ["Earthquake hits coastal region"]

    features = vectorizer.transform(headline)
    prediction = model.predict(features)

    assert prediction[0] in (0, 1)
