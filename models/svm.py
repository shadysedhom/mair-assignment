import os
import joblib
import optuna
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import vstack
from utils.statsRetriever import get_stats


def run_svm_optimization(X_train, X_val, X_test, y_train, y_val, y_test, data_type_name, n_trials=50):
    """
    Performs hyperparameter optimization using Optuna for an SVM classifier.
    This function encapsulates vectorization, optimization, and evaluation.
    Caches the Optuna study to a local file to avoid re-computation.
    """
    print(f"\n--- Running for '{data_type_name}' data ---")

    # Define filename for cached study
    study_filename = f"optuna_study_svm_{data_type_name}.pkl"
    study_filepath = os.path.join("optuna_study_results", study_filename)
    model_type = "SVM"

    # Vectorize the text data, CountVectorizer handles out-of-vocab words by default
    # It does so by ignroring them during transformation
    vectorizer = CountVectorizer()

    # Fit vectroizer only on TRAIN data
    X_train_bow = vectorizer.fit_transform(X_train)

    # Transform val & test data just so they are numerical and compatible w/ SVM
    X_val_bow = vectorizer.transform(X_val)
    X_test_bow = vectorizer.transform(X_test)

    # Caching logic: Check if study exists
    if os.path.exists(study_filepath):
        print(f"Loading existing study for {model_type} on {data_type_name} data from {study_filepath}")
        study = joblib.load(study_filepath)
    else:
        print(f"No cached study found. Creating a new Optuna study for {model_type} on {data_type_name} data.")
        def objective(trial):
            """
            Objective function for Optuna to optimize.
            """
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf']) # Testing linear vs non-linear kernel
            c = trial.suggest_float('C', 1e-2, 1e2, log=True) # Tuning C, which is our regularization param
            
            # Default value (for linear models)
            gamma = 'scale'
            if kernel == 'rbf':
                # For non-linear models tweak gamma to test for different sensitivity levels to individual data points
                gamma = trial.suggest_float('gamma', 1e-2, 1e2, log=True)

            svm = SVC(
                kernel=kernel,
                C=c,
                gamma=gamma,
                random_state=42, # Seed for reproducability
                class_weight='balanced' # Balance class weights to reduce bias
            )
            
            svm.fit(X_train_bow, y_train)   # Fitting model on TRAIN set
            y_pred = svm.predict(X_val_bow) # Tuning on validation set
            accuracy = accuracy_score(y_val, y_pred)
            return accuracy

        # # Suppress Optuna's informational messages
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Run the study to optimize hyperparams
        study = optuna.create_study(direction='maximize')
        print(f"Running Optuna optimization for {data_type_name} data with {n_trials} trials... (This may take a while)")
        study.optimize(objective, n_trials=n_trials)

        # Save the completed study
        joblib.dump(study, study_filepath)
        print(f"Saved new study to {study_filepath}")

    print(f"\nBest parameters found for {data_type_name} data: {study.best_params}")

    # Retrain the best model on the combined training and validation set
    print("Retraining best model on combined train and validation data...")

    # Create a new pipeline with a vectorizer and the best SVM model
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', SVC(random_state=42, class_weight='balanced', **study.best_params))
    ])

    # Combine the raw text data for final training
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    # Fit the entire pipeline on the combined raw text data
    pipeline.fit(X_train_val, y_train_val)

    # Evaluate the pipeline on the test set
    print(f"Evaluating best model on {data_type_name} data test set...")
    y_pred_test = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_test, zero_division=0))

    report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)

    return pipeline, get_stats(report)