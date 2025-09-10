from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Convert text (sentences) into TF-IDF vectors because decision trees do not handle text input directly but need numerical input
# https://machinelearningmastery.com/making-sense-of-text-with-decision-trees/ 
def evaluate_tree(X_train, y_train, X_val, y_val, label):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Accuracy decision tree classifier ({label}): {accuracy:.4f}")
    return accuracy
