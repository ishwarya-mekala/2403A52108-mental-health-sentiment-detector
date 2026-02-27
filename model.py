import re
from typing import List, Tuple

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "at", "by", "for",
    "to", "from", "in", "on", "of", "is", "am", "are", "was", "were", "be", "been",
    "being", "as", "it", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "their", "so", "very",
    "just", "up", "down", "out", "over", "under"
}

_POSITIVE_WORDS = {
    "bless", "blessed", "happy", "calm", "hopeful", "grateful", "support", "supported",
    "friend", "friends", "progress", "proud", "better", "improv", "lift", "safe",
}

_NEGATIVE_WORDS = {
    "sad", "depress", "hopeless", "worthless", "fail", "failing", "anxiet", "panic",
    "attack", "overwhelm", "overwhelmed", "tire", "tired", "cry", "empty", "scared",
    "burden", "hate", "hate myself", "numb", "trapped",
}

_NEUTRAL_WORDS = {
    "okay", "ok", "fine", "manageable", "cope", "coping", "mix", "mixed",
    "bit", "sometimes", "occasionally", "handling", "handle",
}

_stemmer = PorterStemmer()


def _preprocess_text(text: str) -> List[str]:
    """
    Custom preprocessing:
    - lowercasing
    - tokenization (regex-based)
    - stopword removal
    - stemming
    Returns list of processed tokens.
    """
    text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    cleaned_tokens: List[str] = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        stemmed = _stemmer.stem(tok)
        cleaned_tokens.append(stemmed)
    return cleaned_tokens


def _rule_based_sentiment(text: str) -> str | None:
    """
    Simple rule-based override:
    - if clearly positive words and no strong negative words -> 'positive'
    - if clearly negative words and no strong positive words -> 'negative'
    Otherwise, return None and let the ML model decide.
    """
    lowered = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", lowered)
    stemmed_tokens = {_stemmer.stem(t) for t in tokens}

    has_pos = any(p in stemmed_tokens for p in _POSITIVE_WORDS)
    has_neg = any(n in stemmed_tokens for n in _NEGATIVE_WORDS)
    has_neu = any(n in stemmed_tokens for n in _NEUTRAL_WORDS)

    # Handle simple negation like "not good", "not okay", "not fine"
    if "not" in tokens:
        if any(word in tokens for word in ("good", "okay", "ok", "fine", "better", "calm", "happy")):
            return "negative"

    if has_pos and not has_neg:
        return "positive"
    if has_neg and not has_pos:
        return "negative"
    if has_neu and not has_pos and not has_neg:
        return "neutral"
    return None


def load_data(csv_path: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    return df["text"], df["label"]


def build_pipeline() -> Pipeline:
    """
    TF-IDF + Logistic Regression classifier with our custom tokenizer.
    Uses unigrams and bigrams and balances class weights.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=_preprocess_text,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
    )
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )
    return pipeline


def train_model(csv_path: str) -> Tuple[Pipeline, float, str]:
    """
    Train the model and return:
    - fitted pipeline
    - test accuracy
    - text classification report
    """
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return pipeline, acc, report


def predict_sentiment(model: Pipeline, text: str) -> str:
    """
    Predict sentiment label for a single input string.
    """
    rule_label = _rule_based_sentiment(text)
    if rule_label is not None:
        return rule_label
    return model.predict([text])[0]


if __name__ == "__main__":
    model, acc, report = train_model("data/mental_health_dataset.csv")
    print(f"Test accuracy: {acc:.3f}")
    print(report)

