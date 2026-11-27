
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


class NewsTfidf:
    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=20000,
            sublinear_tf=True
        )

    def fit_transform(self, texts):
        word_features = self.word_vectorizer.fit_transform(texts)
        char_features = self.char_vectorizer.fit_transform(texts)
        return hstack([word_features, char_features]).tocsr()

    def transform(self, texts):
        word_features = self.word_vectorizer.transform(texts)
        char_features = self.char_vectorizer.transform(texts)
        return hstack([word_features, char_features]).tocsr()
