from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "report": classification_report(y_test, preds),
    }
    return results


def print_results(results: dict) -> None:
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["report"])
