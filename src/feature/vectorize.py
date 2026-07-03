import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_vectorizer(train_texts, max_features: int = 5000) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vectorizer.fit(train_texts)
    return vectorizer


def create_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Construct an unfitted TF-IDF vectorizer. Fitting happens later,
    inside build_combined_features(), so custom features can be combined
    in the same fit/transform step."""
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

def transform_texts(vectorizer: TfidfVectorizer, texts):
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path: str) -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    sample_texts = ["good movi love it", "bad movi hate it", "great act love"]
    vec = fit_vectorizer(sample_texts, max_features=50)
    matrix = transform_texts(vec, sample_texts)
    print(matrix.shape)
    print(vec.get_feature_names_out())
