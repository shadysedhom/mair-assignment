import os
import joblib
import optuna
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import vstack
from statsRetriever import get_stats

def run_logreg_optimization(X_train, X_val, X_test, y_train, y_val, y_test, data_type_name, n_trials=50):
    """
    Performs hyperparameter optimization using Optuna for a Logistic Regression classifier.
    Includes vectorization, optimization, training, and evaluation.
    Saves and loads studies for reproducibility.
    """
    print(f"\n--- Running Logistic Regression for '{data_type_name}' data ---")

    study_filename = f"optuna_study_logreg_{data_type_name}.pkl"
    model_type = "Logistic Regression"

    # Vectorize text
    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_val_bow = vectorizer.transform(X_val)
    X_test_bow = vectorizer.transform(X_test)

    # Caching: load previous study if available
    if os.path.exists(study_filename):
        print(f"Loading existing study for {model_type} on {data_type_name} from {study_filename}")
        study = joblib.load(study_filename)
    else:
        print(f"No cached study found. Creating a new Optuna study for {model_type} on {data_type_name} data.")

        def objective(trial):
            # Hyperparameter search space
            c = trial.suggest_float("C", 1e-3, 1e2, log=True)   # Regularization strength
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])  
            solver = "liblinear" if penalty == "l1" else "lbfgs"

            logreg = LogisticRegression(
                C=c,
                penalty=penalty,
                solver=solver,
                random_state=42,
                max_iter=1000,
                class_weight="balanced"  # to handle class imbalance
            )

            logreg.fit(X_train_bow, y_train)
            y_pred = logreg.predict(X_val_bow)
            return accuracy_score(y_val, y_pred)

        study = optuna.create_study(direction="maximize")
        print(f"Running Optuna optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)

        joblib.dump(study, study_filename)
        print(f"Saved Optuna study to {study_filename}")

    print(f"\nBest parameters for {data_type_name}: {study.best_params}")

    # Retrain best model on train+val
    print("Retraining Logistic Regression with best params on combined train+val...")
    best_logreg = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",
        **study.best_params,
        solver="liblinear" if study.best_params.get("penalty") == "l1" else "lbfgs"
    )

    X_train_val_bow = vstack([X_train_bow, X_val_bow])
    y_train_val = pd.concat([y_train, y_val])

    best_logreg.fit(X_train_val_bow, y_train_val)

    # Evaluate on test set
    print(f"Evaluating best Logistic Regression on {data_type_name} test set...")
    y_pred_test = best_logreg.predict(X_test_bow)
    print(classification_report(y_test, y_pred_test, zero_division=0))

    report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)

    return best_logreg, get_stats(report)
