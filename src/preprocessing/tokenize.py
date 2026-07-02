import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer (text should already be cleaned)."""
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stemmer.stem(t) for t in tokens]


def process(text: str) -> str:
    """
    Full tokenization pipeline: tokenize -> remove stopwords -> stem.
    Returns a single space-joined string, ready for a vectorizer.
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "i am so happy today the weather is beautiful and i am running fast"
    print(process(sample))
