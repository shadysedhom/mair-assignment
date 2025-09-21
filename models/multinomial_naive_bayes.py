import os
import joblib
import optuna
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from utils.stats_retriever import get_stats


def run_nb_optimization(X_train, y_train, X_val, y_val, X_test, y_test, data_type_name, n_trials=50):
    """
    Performs hyperparameter optimization using Optuna for a Multinomial Naive Bayes classifier.
    Caches the Optuna study to a local file to avoid re-computation.
    """
    print(f"\n--- Running Multinomial Naive Bayes for '{data_type_name}' data ---")

    # Define filename for cached study
    study_filename = f"optuna_study_nb_{data_type_name}.pkl"
    study_filepath = os.path.join("model_tuning", study_filename)
    model_type = "Multinomial Naive Bayes"

    # Vectorize text data using the same settings as before
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_val_bow = vectorizer.transform(X_val)

    # Caching logic: Check if study exists
    if os.path.exists(study_filepath):
        print(f"Loading existing study for {model_type} on {data_type_name} data from {study_filepath}")
        study = joblib.load(study_filepath)
    else:
        print(f"No cached study found. Creating a new Optuna study for {model_type} on {data_type_name} data.")

        def objective(trial):
            """Objective function for Optuna to optimize."""
            # Define hyperparameter search space for the classifier's alpha
            alpha = trial.suggest_float('alpha', 1e-2, 10.0, log=True)

            # Instantiate model
            clf = MultinomialNB(alpha=alpha)
            
            # Fit on (transformed) train set
            clf.fit(X_train_bow, y_train)

            # Optimize on val set
            y_pred = clf.predict(X_val_bow)

            return accuracy_score(y_val, y_pred)

        study = optuna.create_study(direction='maximize')
        print(f"Running Optuna optimization for {data_type_name} data with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)

        # Save the completed study
        joblib.dump(study, study_filepath)
        print(f"Saved new study to {study_filepath}")

    print(f"\nBest parameters found for {data_type_name} data: {study.best_params}")

    # Retrain the best model on the combined training and validation set
    print("Retraining best model on combined train and validation data...")

    # Create a new pipeline with the best found hyperparameters
    pipeline = Pipeline([
        ("bow", CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
        ("clf", MultinomialNB(**study.best_params))
    ])

    # Combine the raw text data for final training
    X_train_val = pd.concat([pd.Series(X_train), pd.Series(X_val)])
    y_train_val = pd.concat([y_train, y_val])

    # Fit the entire pipeline on the combined raw text data
    pipeline.fit(X_train_val, y_train_val)

    # Finally, evaluate the pipeline on the test set
    print(f"Evaluating best model on {data_type_name} data test set...")
    y_pred_test = pipeline.predict(X_test)

    # Generate report dictionary
    report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)

    # Extract and print the concise summary
    accuracy = report['accuracy']
    weighted_f1 = report['weighted avg']['f1-score']
    print(f"Accuracy: {accuracy:.4f}, Weighted F1-Score: {weighted_f1:.4f}")

    return pipeline, get_stats(report)
