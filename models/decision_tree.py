import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pandas as pd
from utils.stats_retriever import get_stats

# Convert text (sentences) into TF-IDF vectors because decision trees do not handle text input directly but need numerical input
# https://machinelearningmastery.com/making-sense-of-text-with-decision-trees/ 
def run_dt_optimization(X_train, y_train, X_val, y_val, X_test, y_test, label):
    print(f"\n--- Running Decision Tree for '{label}' data ---")

    study_filename = f"grid_search_dt_{label}.pkl"
    study_filepath = os.path.join("model_tuning", study_filename)
    model_type = "Decision Tree"

    # Ensure text data has no NaN values
    X_train = pd.Series(X_train).fillna("").astype(str).tolist()
    X_val = pd.Series(X_val).fillna("").astype(str).tolist()
    
    # Merge train + val
    X_combined = X_train + X_val
    y_combined = list(y_train) + list(y_val)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    # Caching: load previous study if available
    if os.path.exists(study_filepath):
        print(f"Loading existing study for {model_type} on {label} from {study_filepath}")
        grid_search = joblib.load(study_filepath)
    else:
        print(f"No cached study found. Creating a new GridSearch study for {model_type} on {label} data.")
        # Define hyperparameters to tune
        param_grid = {
            'clf__max_depth': [1, 5, 10, 15],
            'clf__min_samples_split': [2, 3, 5],
            'clf__criterion': ['gini', 'entropy']
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=KFold(5), scoring='accuracy', n_jobs=-1) # Use all available CPU cores
        grid_search.fit(X_combined, y_combined)

        joblib.dump(grid_search, study_filepath)
        print(f"Saved GridSearch study to {study_filepath}")

    best_model = grid_search.best_estimator_
    print(f"\nBest parameters for {label}: {grid_search.best_params_}")

    print(f"Evaluating best Decision Tree on {label} test set...")
    y_pred = best_model.predict(X_test)
    
    # Generate report dictionary
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Extract and print the concise summary
    accuracy = report['accuracy']
    weighted_f1 = report['weighted avg']['f1-score']
    print(f"\nAccuracy: {accuracy:.4f}, Weighted F1-Score: {weighted_f1:.4f}")

    return best_model, get_stats(report)
