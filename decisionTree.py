from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pandas as pd
# Convert text (sentences) into TF-IDF vectors because decision trees do not handle text input directly but need numerical input
# https://machinelearningmastery.com/making-sense-of-text-with-decision-trees/ 
def evaluate_tree(X_train, y_train, X_val, y_val, X_test, y_test, label):
    
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

    # Define hyperparameters to tune
    param_grid = {
        'clf__max_depth': [1, 3, 5],
        'clf__min_samples_split': [2, 3, 5],
        'clf__criterion': ['gini', 'entropy']
    }

    # If you do not want to use stratified cross-validation, you can use cv=KFold(5) instead, which will create 5 groups without stratification.
    # https://stackoverflow.com/questions/74445334/userwarning-the-least-populated-class-in-y-has-only-1-members-which-is-less-th
    grid_search = GridSearchCV(pipeline, param_grid, cv=KFold(5), scoring='accuracy')
    grid_search.fit(X_combined, y_combined)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy decision tree classifier ({label}): {accuracy:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return best_model