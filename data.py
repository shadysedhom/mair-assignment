import pandas as pd
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