from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from utils.statsRetriever import get_stats


def evaluate_nb(X_train, y_train, X_val, y_val, X_test, y_test, label, alpha=1, min_df=2):

    X_train = pd.Series(X_train).fillna("").astype(str).tolist()
    X_val = pd.Series(X_val).fillna("").astype(str).tolist()

    X_combined = X_train + X_val
    y_combined = list(y_train) + list(y_val)

    pipeline = Pipeline([
        ("bow", CountVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=min_df
        )),
        ("clf", MultinomialNB(alpha=alpha))
    ])

    pipeline.fit(X_combined, y_combined)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate report dictionary
    report = classification_report(y_test, y_pred, digits=3, zero_division=0, output_dict=True)

    # Extract and print the concise summary
    accuracy = report['accuracy']
    weighted_f1 = report['weighted avg']['f1-score']
    print(f"Accuracy: {accuracy:.4f}, Weighted F1-Score: {weighted_f1:.4f}")

    return pipeline, get_stats(report)