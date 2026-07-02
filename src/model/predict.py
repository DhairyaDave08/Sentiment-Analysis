from src.preprocessing.clean_text import clean_text
from src.preprocessing.tokenize import process
from src.features.vectorize import load_vectorizer
from src.models.train import load_model

LABEL_MAP = {0: "negative", 4: "positive"}  # adjust to match your dataset's label scheme


def predict_sentiment(text: str, model_path: str, vectorizer_path: str) -> str:
    vectorizer = load_vectorizer(vectorizer_path)
    model = load_model(model_path)

    cleaned = clean_text(text)
    processed = process(cleaned)

    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]

    return LABEL_MAP.get(prediction, str(prediction))


if __name__ == "__main__":
    result = predict_sentiment(
        "I absolutely loved this, best day ever!",
        model_path="models/sentiment_model.pkl",
        vectorizer_path="models/vectorizer.pkl",
    )
    print(result)
