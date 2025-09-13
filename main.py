import os

import BaselineSystems as baseline
from data import load_and_preprocess_data, split_data
from svm import run_svm_optimization
from logistic_regression import run_logreg_optimization
from decisionTree import evaluate_tree
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":
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


    #* ---------- TODO: MAJORITY VOTE (NAIVE) BASELINE ------------ just predict the most common label everytime

    print("\n" + "-"*50 + "\nBaselines\n" + "-"*50)

    resultBaseLineMajority = baseline.calculate_majority_label_accuracy(df_with_duplicates.values.tolist())
    print("Majority Baseline Accuracy (duplicated):", resultBaseLineMajority)

    resultBaseLineMajority = baseline.calculate_majority_label_accuracy(df_without_duplicates.values.tolist())
    print("Majority Baseline Accuracy (deduplicated):", resultBaseLineMajority)

#* ---------- TODO: MANUAL RULE BASED BASELINE -------- iterate till it scores over 80% (no stats/ml just manual rules)

    resultBaseLine = baseline.calculate_accuracy(df_with_duplicates.values.tolist())
    print("Rule-Based Accuracy (duplicated):", resultBaseLine)

    resultBaseLine = baseline.calculate_accuracy(df_without_duplicates.values.tolist())
    print("Rule-Based Accuracy (deduplicated):", resultBaseLine)




    #* --------- TODO: Build Classifier 1 (Bram) --------- Make 2 versions of your model:
    #* --------- one with the original data and split, one with the deduplicated data and split.   
    #* -- Use bag of words representation and handle out of vocabulary words --
    print("\n" + "-"*50)
    print("Classifier 2: Logistic Regression")
    print("-"*50)

    # Run with original data
    run_logreg_optimization(
        X_train_orig, X_val_orig, X_test_orig,
        y_train_orig, y_val_orig, y_test_orig,
        "original"
    )

    # Run with deduplicated data
    run_logreg_optimization(
        X_train_dedup, X_val_dedup, X_test_dedup,
        y_train_dedup, y_val_dedup, y_test_dedup,
        "deduplicated"
    )

    print("-"*50)


    #* --------- TODO: Build Classifier 2 (Lenny) -------- Make 2 versions of your model:
    #* --------- one with the original data and split, one with the deduplicated data and split.   
    #* -- Use bag of words representation and handle out of vocabulary words --



    #* --------- Classifier 3: Support Vector Machine (SVM) ------------
    print("\n" + "-"*50 + "\nClassifier 3: Support Vector Machine\n" + "-"*50)

    # Call the function for the original data
    svm = run_svm_optimization(
        X_train_orig, X_val_orig, X_test_orig,
        y_train_orig, y_val_orig, y_test_orig,
        "original"
    )

    # Call the function for the deduplicated data
    run_svm_optimization(
        X_train_dedup, X_val_dedup, X_test_dedup,
        y_train_dedup, y_val_dedup, y_test_dedup,
        "deduplicated"
    )
    
    print("-"*50)



    #* ---- TODO: After training, testing, and reporting performance, 
    #* ---- the program should offer a prompt to enter a new sentence and classify this sentence,
    #* ---- and repeat the prompt until the user exits.  
    #* !! Convert ALL user input to lowercase !!

    #* --------- TODO: Build Classifier 4 (Dirk-Jan) -------- Make 2 versions of your model:
    #* --------- one with the original data and split, one with the deduplicated data and split.   
    #* -- Use bag of words representation and handle out of vocabulary words --

    print("Classifier 4: Decision Tree\n" + "-"*50)

    decision_tree_model = evaluate_tree(X_train_dedup, y_train_dedup, X_val_dedup, y_val_dedup, X_test_dedup, y_test_dedup, "deduplicated")
    evaluate_tree(X_train_orig, y_train_orig, X_val_orig, y_val_orig, X_test_orig, y_test_orig, "original")

    print("-"*50)

    #* ---- TODO: After training, testing, and reporting performance, 
    #* ---- the program should offer a prompt to enter a new sentence and classify this sentence,
    #* ---- and repeat the prompt until the user exits.  
    #* !! Convert ALL user input to lowercase !!




    #* ------ TODO: EVALUATION (Dirk-Jan) ---------

    print("evaluation on custom test set:")
    print("-"*50)

    custom_test_set = [
        ("phonenumer please!", "request"),
        ("you are not wrong ", "affirm")
    ]
    X_test = [item[0] for item in custom_test_set]
    y_test = [item[1] for item in custom_test_set]

    y_pred_decision_tree_custom = decision_tree_model.predict(X_test)
    print("Decision Tree (input output):", y_test, y_pred_decision_tree_custom)

    print("-"*50)