
import joblib
from pathlib import Path


def get_model_and_vectorizer():
    model = joblib.load(Path("models/linearsvm.joblib"))
    vectorizer = joblib.load(Path("models/tfidf_vectorizer.joblib"))
    return model, vectorizer


def test_model_handles_empty_feature_text():
    model, vectorizer = get_model_and_vectorizer()
    features = vectorizer.transform([""])

    prediction = model.predict(features)

    assert prediction[0] in (0, 1)


def test_model_handles_unseen_words():
    model, vectorizer = get_model_and_vectorizer()
    features = vectorizer.transform(
        ["Zorblax quantum flux research announcement"]
    )

    prediction = model.predict(features)

    assert prediction[0] in (0, 1)


def test_model_handles_long_news_text():
    model, vectorizer = get_model_and_vectorizer()
    text = "Scientists publish a new research report. " * 100

    features = vectorizer.transform([text])
    prediction = model.predict(features)

    assert len(prediction) == 1


def test_model_predictions_are_deterministic():
    model, vectorizer = get_model_and_vectorizer()
    features = vectorizer.transform(
        ["Central bank announces new economic policy"]
    )

    first = model.predict(features)
    second = model.predict(features)

    assert first[0] == second[0]


def test_vectorizer_output_is_sparse():
    _, vectorizer = get_model_and_vectorizer()
    features = vectorizer.transform(
        ["Government announces new economic policy"]
    )

    assert hasattr(features, "tocsr")
    assert features.shape[0] == 1
