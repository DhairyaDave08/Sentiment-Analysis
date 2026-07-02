import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "naive_bayes": lambda: MultinomialNB(),
    "svm": lambda: LinearSVC(),
}


def train_model(X_train, y_train, model_name: str = "logistic_regression"):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}"
        )
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
