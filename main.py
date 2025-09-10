import os
import pandas as pd
import BaselineSystems as baseline
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    """
    Loads data from the specified file into a pandas DataFrame.
    Assumes each line is in the format: 'dialog_act [space] utterance_content'
    Converts all text to lowercase, handles multiple dialog acts by taking only the first one,
    and handles missing/null values by dropping malformed rows.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Convert the entire line to lowercase
            line = line.lower().strip()

            # Skip empty lines
            if not line:
                continue

            parts = line.split(' ', 1)
            if len(parts) < 2:
                # Skip lines that do not conform to the expected format (missing utterance content)
                continue

            dialog_act = parts[0]
            utterance = parts[1]

            data.append({'dialog_act': dialog_act, 'utterance': utterance}) # Add entry

    # Convert to dataframe
    df = pd.DataFrame(data)

    # Since some acts have no label, and some utterances are completely unintelligible, we need to handle those.
    # Calculate and print the number and percentage of 'null' and 'unintelligible' dialog acts
    null_count = df[df['dialog_act'] == 'null'].shape[0]
    unintelligible_count = df[df['utterance'] == 'unintelligible'].shape[0]
    total_rows = df.shape[0]

    print("Total rows BEFORE handling missing values: ", total_rows)
    print(f"Number of 'null' dialog acts: {null_count}")
    print(f"Percentage of null dialog acts: {(null_count / total_rows) * 100:.2f}%")
    print(f"Number of 'unintelligible' utterances: {unintelligible_count}")
    print(f"Percentage of unintelligible utterances: {(unintelligible_count / total_rows) * 100:.2f}%")
    print(f"Dropping missing values...")

    # Drop rows where dialog_act is 'null' or utterance is 'unintelligible'
    df = df[df['dialog_act'] != 'null']
    df = df[df['utterance'] != 'unintelligible']
    print("Total rows AFTER handling missing values: ", df.shape[0])

    print("-"*50 + f"\nLoaded and preprocessed {len(df)} rows.")
    return df

def split_data(df, test_size=0.15, val_size=0.10, seed=42):
    """
    Splits the data into training, validation, and test sets.
    Approach: 85% train, 15% test. Then split train such that 10% of total data is used for val set.
    """
    # Split into 85% train and 15% test
    X_train, X_test, y_train, y_test = train_test_split(
        df['utterance'], df['dialog_act'], test_size=test_size, random_state=seed
    )

    # Calculate the proportion of the validation set from the new training set
    # val_size (0.10) / (1 - test_size (0.15)) = 0.10 / 0.85
    val_from_train_size = val_size / (1 - test_size)

    # Split the training set further into actual training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_from_train_size, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


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

resultBaseLineMajority = baseline.calculate_majority_label_accuracy(df_with_duplicates.values.tolist())
print("Majority Baseline Accuracy:", resultBaseLineMajority)

#* ---------- TODO: MANUAL RULE BASED BASELINE -------- iterate till it scores over 80% (no stats/ml just manual rules)

resultBaseLine = baseline.calculate_accuracy(df_without_duplicates.values.tolist())
print("Rule-Based Accuracy:", resultBaseLine)




#* --------- TODO: Build Classifier 1 (Bram) --------- Make 2 versions of your model:
#* --------- one with the original data and split, one with the deduplicated data and split.   
#* -- Use bag of words representation and handle out of vocabulary words --



#* --------- TODO: Build Classifier 2 (Lenny) -------- Make 2 versions of your model:
#* --------- one with the original data and split, one with the deduplicated data and split.   
#* -- Use bag of words representation and handle out of vocabulary words --



#* --------- TODO: Build Classifier 3 (Shady) -------- Make 2 versions of your model:
#* --------- one with the original data and split, one with the deduplicated data and split.   
#* -- Use bag of words representation and handle out of vocabulary words --





#* ---- TODO: After training, testing, and reporting performance, 
#* ---- the program should offer a prompt to enter a new sentence and classify this sentence,
#* ---- and repeat the prompt until the user exits.  
#* !! Convert ALL user input to lowercase !!




#* ------ TODO: EVALUATION (Dirk-Jan) ---------



