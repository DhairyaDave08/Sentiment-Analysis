# 🐦 Twitter Sentiment Analysis

A classical ML sentiment classifier for tweets — no deep learning required. ✨
Given a tweet, the model predicts whether it's **positive** 🙂 or **negative** 🙁,
using TF-IDF features and a Logistic Regression / Naive Bayes / SVM classifier.

Built on the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
dataset (1.6M labeled tweets). 📊

## 📁 Repo structure

```
Sentiment-Analysis/
├── Executive_files/
│   ├── sentiment_model.pkl       # trained model
│   ├── vectorizer.pkl             # trained TF-IDF vectorizer
│   └── 03_Execution.ipynb          # final run — loads model, takes tweet input, predicts sentiment
├── Notebook/
│   ├── 01_EDA.ipynb                # data exploration, cleaning checks, word frequency
│   └── 02_Model.ipynb               # training pipeline — vectorize, train, evaluate, save
├── src/                              # core pipeline code (cleaning, tokenizing, vectorizing, training)
├── .gitignore
├── LICENSE
└── README.md
```

**📌 Note on data:** the raw dataset isn't stored in this repo (1.6M rows is too
large to version-control cleanly). It's pulled at runtime via `kagglehub`
directly in the notebooks — see the "Run it yourself" section below.

## 🚀 Quickstart — try the model

The fastest way to test it is `Executive_files/03_Execution.ipynb`, which:
1. 📥 Loads the trained model + vectorizer from `Executive_files/`
2. ⌨️ Takes a tweet as input
3. 🧠 Prints the predicted sentiment

Open `02_Model.ipynb` in Colab, run all cells, and enter a tweet when prompted.

```python
# Example
tweet = "I absolutely loved this, best day ever!"
predict_sentiment(tweet)
# -> "positive"
```

## 📈 Results

Trained and compared on the full 1.6M-tweet Sentiment140 dataset (80/20 split):

| Model | Accuracy |
|---|---|
| **Logistic Regression** ✅ | 77.11% |
| SVM | 77.07% |
| Naive Bayes | 75.44% |

Logistic Regression performed best (marginally ahead of SVM) and is the model
saved in `Executive_files/`.

## ⚙️ How it works

1. **Cleaning** 🧹 — strip URLs, @mentions, hashtags (keep the word), and non-alphabetic characters
2. **Tokenizing** ✂️ — remove stopwords, apply Porter stemming
3. **Vectorizing** 🔢 — TF-IDF (unigrams + bigrams)
4. **Classifying** 🏷️ — Logistic Regression / Naive Bayes / Linear SVM (best model saved to `Executive_files/`)

## 🔁 Run the full pipeline yourself

If you want to retrain from scratch rather than using the saved model:

1. Open `Notebook/01_EDA.ipynb` — pulls Sentiment140 via `kagglehub`,
   explores class balance and tweet length, and sanity-checks the cleaning pipeline
2. Open `Notebook/02_Model.ipynb` — runs the training steps (vectorize → train →
   evaluate → save); model and vectorizer are saved as `.pkl` files
3. Use `Executive_files/03_Execution.ipynb` to load your newly trained model and test it

## 🔮 Future Scope

- 🔀 **Negation handling** — explicit rules or bigram-aware features so "not good" isn't scored like "good"
- 🎭 **Sarcasm detection** — would need context-aware models (RNNs/transformers) beyond bag-of-words
- ⚖️ **Aspect-level sentiment** — split mixed-sentiment tweets (e.g. "great acting, bad plot") into separate scores per aspect instead of one averaged label
- 🧠 **Word embeddings (Word2Vec/GloVe)** — capture semantic similarity between words instead of treating them as independent TF-IDF features
- 🤖 **Transformer-based models (BERT)** — for a stronger accuracy ceiling once moving past classical ML
- 🌐 **Multi-class sentiment** — extend beyond positive/negative to include neutral, and possibly emotion categories (joy, anger, etc.)
- 📡 **Live tweet stream integration** — hook into a live API/feed for real-time sentiment tracking instead of static batch predictions
- 🖥️ **Web app / API deployment** — wrap the model in a simple Flask/Streamlit app or REST API for easier public use

## 📄 License

See [LICENSE](LICENSE).

---

⭐ **If you found this project useful or interesting, consider starring the repo!**
I'm always open to good collaboration — feel free to open an issue, submit a PR,
or reach out if you'd like to contribute or build on this together. 🤝
