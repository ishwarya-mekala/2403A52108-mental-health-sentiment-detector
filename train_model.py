import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data") / "mental_health_data.csv"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "mental_health_tfidf_model.joblib"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please place a CSV with columns 'text' and 'label' there."
        )
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"])
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Evaluation on held-out test set:")
    print(classification_report(y_test, y_pred))

    return pipeline


def main() -> None:
    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} examples.")

    print("Training TF-IDF + LogisticRegression model...")
    pipeline = train_and_evaluate(df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

