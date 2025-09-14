import os

import BaselineSystems as baseline
from data import load_and_preprocess_data, split_data
from statsRetriever import SystemsOverview
from svm import run_svm_optimization
from logistic_regression import run_logreg_optimization
from decisionTree import evaluate_tree


if __name__ == "__main__":

    systems_overview = SystemsOverview()

    # ---- CONSTANTS ------
    DASHED_LINE = "-" * 100

    # Start  of script
    data_filepath = os.path.join(os.path.dirname(__file__), 'dialog_acts.dat')

    # Load and preprocess the data
    print(f"Loading and preprocessing data... \n")
    df_with_duplicates = load_and_preprocess_data(data_filepath)

    # Remove duplicates
    df_without_duplicates = df_with_duplicates.drop_duplicates(subset=['utterance'])
    print(f"\nCreated a copy of the data without duplicates. Total rows: {df_without_duplicates.shape[0]}")

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


    #* -------------       MAJORITY VOTE (NAIVE) BASELINE       ---------------
    # here we just predict the most occuring label in the data (everytime)

    print("\n" + DASHED_LINE + "\nBaselines\n" + DASHED_LINE)

    resultBaseLineMajority = baseline.calculate_majority_label_accuracy(df_with_duplicates.values.tolist())
    print("Majority Baseline Accuracy (duplicated):", resultBaseLineMajority)

    resultBaseLineMajority = baseline.calculate_majority_label_accuracy(df_without_duplicates.values.tolist())
    print("Majority Baseline Accuracy (deduplicated):", resultBaseLineMajority)

    #* -------------       MANUAL RULE BASED BASELINE        -----------------
    # Comprised of manualy found rules

    resultBaseLine = baseline.calculate_accuracy(df_with_duplicates.values.tolist())
    print("Rule-Based Accuracy (duplicated):", resultBaseLine)

    resultBaseLine = baseline.calculate_accuracy(df_without_duplicates.values.tolist())
    print("Rule-Based Accuracy (deduplicated):", resultBaseLine)


    #* --------- Classifier 1: Logistic Regression ------------
    print("\n" + DASHED_LINE + "\nClassifier 1: Logistic Regression\n" + DASHED_LINE)

    # Run with original data
    logreg, logreg_metrics_original = run_logreg_optimization(
        X_train_orig, X_val_orig, X_test_orig,
        y_train_orig, y_val_orig, y_test_orig,
        "original"
    )

    # Run with deduplicated data
    logreg, logreg_metrics_deduplicated = run_logreg_optimization(
        X_train_dedup, X_val_dedup, X_test_dedup,
        y_train_dedup, y_val_dedup, y_test_dedup,
        "deduplicated"
    )

    systems_overview.add_system_results("Logistic Regression", logreg_metrics_original, logreg_metrics_deduplicated)

    #* --------- TODO: Build Classifier 2 (Lenny) -------- Make 2 versions of your model:
    #* --------- one with the original data and split, one with the deduplicated data and split.   
    #* -- Use bag of words representation and handle out of vocabulary words --



    #* --------- Classifier 3: Support Vector Machine (SVM) ------------
    print("\n" + DASHED_LINE + "\nClassifier 3: Support Vector Machine\n" + DASHED_LINE)

    # Call the function for the original data
    svm, svm_metrics_original = run_svm_optimization(
        X_train_orig, X_val_orig, X_test_orig,
        y_train_orig, y_val_orig, y_test_orig,
        "original"
    )

    # Call the function for the deduplicated data
    svm, svm_metrics_deduplicated = run_svm_optimization(
        X_train_dedup, X_val_dedup, X_test_dedup,
        y_train_dedup, y_val_dedup, y_test_dedup,
        "deduplicated"
    )

    systems_overview.add_system_results("SVM", svm_metrics_original, svm_metrics_deduplicated)
    
    #* --------- Classifier 4: Decision Tree ------------
    print("\n" + DASHED_LINE +"\nClassifier 4: Decision Tree\n" + DASHED_LINE)

    # Once for the original data
    decision_tree_model_original, decision_tree_metrics_original = evaluate_tree(X_train_orig, y_train_orig, X_val_orig, y_val_orig, X_test_orig, y_test_orig, "original")

    # Once for deduplicated data
    decision_tree_model, decision_tree_metrics_deduplicated = evaluate_tree(X_train_dedup, y_train_dedup, X_val_dedup, y_val_dedup, X_test_dedup, y_test_dedup, "deduplicated")

    systems_overview.add_system_results("Decision Tree", decision_tree_metrics_original, decision_tree_metrics_deduplicated)

    #* ---- TODO: After training, testing, and reporting performance, 
    #* ---- the program should offer a prompt to enter a new sentence and classify this sentence,
    #* ---- and repeat the prompt until the user exits.  
    #* !! Convert ALL user input to lowercase !!




    #* ------ TODO: EVALUATION (Dirk-Jan) ---------

    print("evaluation on custom test set:")
    print(DASHED_LINE)

    custom_test_set = [
        ("phonenumer please!", "request"),
        ("you are not wrong ", "affirm")
    ]
    X_test = [item[0] for item in custom_test_set]
    y_test = [item[1] for item in custom_test_set]

    y_pred_decision_tree_custom = decision_tree_model.predict(X_test)
    print("Decision Tree (input output):", y_test, y_pred_decision_tree_custom)

    print(DASHED_LINE)

    print("\nFinal results summary:")
    systems_overview.print_results_table()