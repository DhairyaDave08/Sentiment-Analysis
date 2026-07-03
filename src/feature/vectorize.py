

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """
    Construct an UNFITTED TF-IDF vectorizer.

    Use this (instead of fit_vectorizer) when you plan to combine TF-IDF
    with custom features via build_combined_features(), since that function
    calls .fit_transform() / .transform() on this vectorizer itself.
    """
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))


def fit_vectorizer(train_texts, max_features: int = 5000) -> TfidfVectorizer:
    """
    Construct AND fit a TF-IDF vectorizer on training text in one step.

    Use this for TF-IDF-only workflows (no custom features).
    """
    vectorizer = create_vectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts):
    """Transform text into TF-IDF features using an already-fitted vectorizer."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path: str) -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    sample_texts = ["good movi love it", "bad movi hate it", "great act love"]

    # TF-IDF-only path
    vec = fit_vectorizer(sample_texts, max_features=50)
    matrix = transform_texts(vec, sample_texts)
    print("TF-IDF-only shape:", matrix.shape)
    print(vec.get_feature_names_out())

    # Unfitted vectorizer path (for use with build_combined_features)
    unfitted_vec = create_vectorizer(max_features=50)
    print("Unfitted vectorizer created:", unfitted_vec)
