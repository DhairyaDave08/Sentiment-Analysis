import re
import numpy as np
from scipy.sparse import hstack, csr_matrix


NEGATION_WORDS = {
    "not", "no", "never", "cant", "cannot", "wont", "dont",
    "isnt", "arent", "wasnt", "werent", "didnt", "doesnt",
    "hasnt", "havent", "wouldnt", "shouldnt", "couldnt",
}


def negation_count(cleaned_text: str) -> int:
    """Count negation words in already-cleaned/tokenized text."""
    tokens = cleaned_text.lower().split()
    return sum(1 for t in tokens if t in NEGATION_WORDS)


def exclamation_count(raw_text: str) -> int:
    return raw_text.count("!")


def question_count(raw_text: str) -> int:
    return raw_text.count("?")


def caps_ratio(raw_text: str) -> float:
    letters = [c for c in raw_text if c.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters)


def elongation_count(raw_text: str) -> int:
    return len(re.findall(r"(.)\1{2,}", raw_text))


def word_count(raw_text: str) -> int:
    return len(raw_text.split())


def extract_custom_features(raw_text: str, cleaned_text: str) -> np.ndarray:
   
    return np.array([
        negation_count(cleaned_text),
        exclamation_count(raw_text),
        question_count(raw_text),
        caps_ratio(raw_text),
        elongation_count(raw_text),
        word_count(raw_text),
    ], dtype=np.float64)


def extract_custom_features_batch(raw_texts, cleaned_texts) -> np.ndarray:

    return np.array([
        extract_custom_features(raw, cleaned)
        for raw, cleaned in zip(raw_texts, cleaned_texts)
    ])


def build_combined_features(df, vectorizer, fit: bool = False, text_col: str = "text",
                             processed_col: str = "processed"):
  
    if fit:
        tfidf_matrix = vectorizer.fit_transform(df[processed_col])
    else:
        tfidf_matrix = vectorizer.transform(df[processed_col])

    custom_feats = extract_custom_features_batch(df[text_col], df[processed_col])

    combined = hstack([tfidf_matrix, csr_matrix(custom_feats)])
    return combined


def get_feature_names(vectorizer) -> list[str]:

    tfidf_names = list(vectorizer.get_feature_names_out())
    custom_names = [
        "negation_count",
        "exclamation_count",
        "question_count",
        "caps_ratio",
        "elongation_count",
        "word_count",
    ]
    return tfidf_names + custom_names


if __name__ == "__main__":
    import pandas as pd

    sample_df = pd.DataFrame({
        "text": [
            "This is NOT good at all!!!",
            "I loooove this so much",
            "meh, it's fine I guess",
        ],
        "processed": [
            "not good",
            "loooove much",
            "meh fine guess",
        ],
    })

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=50)

    combined = build_combined_features(sample_df, vec, fit=True)
    print("Combined shape:", combined.shape)
    print("Feature names (last 6 are custom):", get_feature_names(vec)[-6:])
    print("Custom feature values:\n", combined[:, -6:].toarray())
