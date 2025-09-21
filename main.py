import os

from data.data import load_and_preprocess_data, split_data
from utils.stats_retriever import SystemsOverview
from sklearn.metrics import accuracy_score
from cli import start_cli

import models.baseline_systems as baseline
from models.logistic_regression import run_logreg_optimization
from models.multinomial_naive_bayes import run_nb_optimization
from models.svm import run_svm_optimization
from models.decision_tree import run_dt_optimization

from dialogue_system.keyword_searcher import RestaurantSearcher
from dialogue_system.restaurant_manager import RestaurantManager
from dialogue_system.restaurant_reader import RestaurantReader


if __name__ == "__main__":

    systems_overview = SystemsOverview()

    # ---- CONSTANTS ------
    DASHED_LINE = "-" * 100

    # Start  of script
    data_filepath = os.path.join(os.path.dirname(__file__), './data/dialog_acts.dat')

    # Load and preprocess the data
    print(f"Loading and preprocessing data... \n")
    df_with_duplicates = load_and_preprocess_data(data_filepath)

    # Remove duplicates
    df_without_duplicates = df_with_duplicates.drop_duplicates(subset=['utterance'])
    print(f"Created a copy of the data without duplicates. Total rows: {df_without_duplicates.shape[0]}")

    # Split the original data (with duplicates) ---------- Here is the data we use WITH duplicates ----------
    print("\nSplitting original data (with duplicates)...")
    (X_train_orig, X_val_orig, X_test_orig,
     y_train_orig, y_val_orig, y_test_orig) = split_data(df_with_duplicates)
    print(f"Original data split: Train={len(X_train_orig)}, Val={len(X_val_orig)}, Test={len(X_test_orig)}")

    # Split the deduplicated data ---------- Data WITHOUT duplicates ----------
    print("\nSplitting deduplicated data...")
    (X_train_dedup, X_val_dedup, X_test_dedup,
     y_train_dedup, y_val_dedup, y_test_dedup) = split_data(df_without_duplicates)
    print(f"Deduplicated data split: Train={len(X_train_dedup)}, Val={len(X_val_dedup)}, Test={len(X_test_dedup)}")


    #* ------------- Baselines ---------------
    print("\n" + DASHED_LINE + "\nBaselines\n" + DASHED_LINE)

    # --- Majority Baseline ---
    print("--- Majority Baseline ---")

    # On original data
    majority_model_orig = baseline.MajorityBaseline()
    majority_model_orig.fit(y_train_orig)
    y_pred_maj_orig = majority_model_orig.predict(X_test_orig)
    accuracy_maj_orig = accuracy_score(y_test_orig, y_pred_maj_orig)
    print(f"Majority Baseline Accuracy (original): {accuracy_maj_orig:.4f}")

    # On deduplicated data
    majority_model_dedup = baseline.MajorityBaseline()
    majority_model_dedup.fit(y_train_dedup)
    y_pred_maj_dedup = majority_model_dedup.predict(X_test_dedup)
    accuracy_maj_dedup = accuracy_score(y_test_dedup, y_pred_maj_dedup)
    print(f"Majority Baseline Accuracy (deduplicated): {accuracy_maj_dedup:.4f}")

    # --- Rule-Based Baseline ---
    print("\n--- Rule-Based Baseline ---")
    # On original data
    rule_model_orig = baseline.RuleBasedBaseline()
    y_pred_rule_orig = rule_model_orig.predict(X_test_orig)
    accuracy_rule_orig = accuracy_score(y_test_orig, y_pred_rule_orig)
    print(f"Rule-Based Baseline Accuracy (original): {accuracy_rule_orig:.4f}")

    # On deduplicated data
    rule_model_dedup = baseline.RuleBasedBaseline()
    y_pred_rule_dedup = rule_model_dedup.predict(X_test_dedup)
    accuracy_rule_dedup = accuracy_score(y_test_dedup, y_pred_rule_dedup)
    print(f"Rule-Based Baseline Accuracy (deduplicated): {accuracy_rule_dedup:.4f}")


    #* --------- Classifier 1: Logistic Regression ------------
    print("\n" + DASHED_LINE + "\nClassifier 1: Logistic Regression\n" + DASHED_LINE)

    # Run with original data
    logreg_original_model, logreg_metrics_original = run_logreg_optimization(
        X_train_orig, X_val_orig, X_test_orig,
        y_train_orig, y_val_orig, y_test_orig,
        "original"
    )

    # Run with deduplicated data
    logreg_deduplicated_model, logreg_metrics_deduplicated = run_logreg_optimization(
        X_train_dedup, X_val_dedup, X_test_dedup,
        y_train_dedup, y_val_dedup, y_test_dedup,
        "deduplicated"
    )

    systems_overview.add_system_results("Logistic Regression", logreg_metrics_original, logreg_metrics_deduplicated)

    #* --------- Classifier 2: Multinomial Naive Bayes ------------
    print("\n" + DASHED_LINE + "\nClassifier 2: Multinomial Naive Bayes\n" + DASHED_LINE)

    # On the original data
    multinomial_nb_model_original, multinomial_nb_metrics_original = run_nb_optimization(
        X_train_orig, y_train_orig, X_val_orig, y_val_orig, X_test_orig, y_test_orig, "original"
    )

    # On the deduplicated data
    multinomial_nb_model_deduplicated, multinomial_nb_metrics_deduplicated = run_nb_optimization(
        X_train_dedup, y_train_dedup, X_val_dedup, y_val_dedup, X_test_dedup, y_test_dedup, "deduplicated"
    )

    systems_overview.add_system_results("Multinomial Naive Bayes", multinomial_nb_metrics_original, multinomial_nb_metrics_deduplicated)

    #* --------- Classifier 3: Support Vector Machine (SVM) ------------
    print("\n" + DASHED_LINE + "\nClassifier 3: Support Vector Machine\n" + DASHED_LINE)

    # Call the function for the original data
    svm_original_model, svm_metrics_original = run_svm_optimization(
        X_train_orig, X_val_orig, X_test_orig,
        y_train_orig, y_val_orig, y_test_orig,
        "original"
    )

    # Call the function for the deduplicated data
    svm_deduplicated_model, svm_metrics_deduplicated = run_svm_optimization(
        X_train_dedup, X_val_dedup, X_test_dedup,
        y_train_dedup, y_val_dedup, y_test_dedup,
        "deduplicated"
    )

    systems_overview.add_system_results("SVM", svm_metrics_original, svm_metrics_deduplicated)
    
    #* --------- Classifier 4: Decision Tree ------------
    print("\n" + DASHED_LINE +"\nClassifier 4: Decision Tree\n" + DASHED_LINE)

    # Once for the original data
    decision_tree_model_original, decision_tree_metrics_original = run_dt_optimization(X_train_orig, y_train_orig, X_val_orig, y_val_orig, X_test_orig, y_test_orig, "original")

    # Once for deduplicated data
    decision_tree_model_deduplicated, decision_tree_metrics_deduplicated = run_dt_optimization(X_train_dedup, y_train_dedup, X_val_dedup, y_val_dedup, X_test_dedup, y_test_dedup, "deduplicated")

    systems_overview.add_system_results("Decision Tree", decision_tree_metrics_original, decision_tree_metrics_deduplicated)

    #* ------ Evaluation ---------
    print("\nEvaluation on custom test set:\n" + DASHED_LINE)

    custom_test_set = [
        ("phonenumer please!", "request"),
        ("you are not wrong ", "affirm")
    ]
    X_test = [item[0] for item in custom_test_set]
    y_test = [item[1] for item in custom_test_set]

    y_pred_decision_tree_custom = decision_tree_model_deduplicated.predict(X_test)
    print("Decision Tree (input output):", y_test, y_pred_decision_tree_custom)

    print("\n" + DASHED_LINE + "\nFinal results summary:")
    systems_overview.print_results_table()

    #*---------------------- Interactive Classifier --------------------------
    # Store of the trained models (on DEDUPLICATED data)
    models = {
        "Logistic Regression": logreg_deduplicated_model,
        "Multinomial Naive Bayes": multinomial_nb_model_deduplicated,
        "SVM": svm_deduplicated_model,
        "Decision Tree": decision_tree_model_deduplicated
    }

    restaurant_searcher = RestaurantSearcher(RestaurantManager(RestaurantReader(os.path.join(os.path.dirname(__file__), './data/restaurant_info.csv')).read_restaurants()))
    print("\n" + DASHED_LINE + "\n keyword search example:")
    restaurant_searcher.search("I want a cheap restaurant in the north that serves chines food", "pricerange")
    restaurant_searcher.search("I want a cheap restaurant in the north that serves chines food", "area")
    restaurant_searcher.search("I want a cheap restaurant in the north that serves chines food", "food")
    print(restaurant_searcher.unique_pricerange, restaurant_searcher.unique_area, restaurant_searcher.unique_food)

    # Start simple CLI
    start_cli(models)
