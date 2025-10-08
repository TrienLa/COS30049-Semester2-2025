import os
import pandas as pd
import sys
from data_processing import load_data, data_clean_up, data_preprocessing



def preprocess_train_dataset(path):
    # Read the training dataset from the given path
    df = pd.read_csv(path)
    
    # Some versions call the text column "message" instead of "text"
    if 'text' not in df.columns and 'message' in df.columns:
        df.rename(columns={'message': 'text'}, inplace=True)
    
    # Convert the 'label' column (ham/spam) into numeric values: 0 = ham, 1 = spam
    if 'label' in df.columns:
        df['spam'] = df['label'].map({'ham': 0, 'spam': 1})
        df.drop('label', axis=1, inplace=True)

    # Drop leftover or useless columns that might exist in some datasets
    drop_cols = [col for col in ['Unnamed: 0', 'label_num'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Remove any rows that don't have both text and spam values
    df.dropna(subset=['text', 'spam'], inplace=True)
    
    # Clean up the index after removing rows
    df.reset_index(drop=True, inplace=True)
    
    return df


def preprocess_email_spam_dataset(path):
    # Read the email_spam.csv file
    df = pd.read_csv(path)

    # Sometimes there's a "title" column we don't need — drop it if it exists
    if 'title' in df.columns:
        df.drop('title', axis=1, inplace=True)

    # The dataset may label spam as text ("spam" / "not spam") — turn that into 1s and 0s
    if 'type' in df.columns:
        df['spam'] = df['type'].map({'spam': 1, 'not spam': 0})
        df.drop('type', axis=1, inplace=True)

    # Drop rows that are missing text or spam values
    df.dropna(subset=['text', 'spam'], inplace=True)
    
    # Reset index so everything stays in order after cleaning
    df.reset_index(drop=True, inplace=True)
    
    return df

def preprocess_enron_dataset(path):
    """
    Preprocesses the Enron Spam/Ham dataset to match the structure of emails.csv.
    
    Expected columns:
        - 'Message:' (ID)
        - 'Subject:' (subject line)
        - 'Message:' (body text)
        - 'Spam/Ham:' ('spam' or 'ham')
        - 'Date:' (date string)

    Returns:
        DataFrame with columns ['text', 'spam'].
    """
    # Load the CSV file
    df = pd.read_csv(path, encoding='latin1')

    # Normalize column names to lowercase and strip whitespace
    df.columns = [c.strip().lower().replace(':', '') for c in df.columns]

    # Identify possible text columns
    subject_col = next((c for c in df.columns if 'subject' in c), None)
    message_cols = [c for c in df.columns if 'message' in c]
    label_col = next((c for c in df.columns if 'spam' in c or 'ham' in c), None)

    if not subject_col or not message_cols or not label_col:
        raise KeyError("One or more expected columns ('Subject', 'Message', 'Spam/Ham') not found in dataset.")

    # Pick the message column that’s textual, not numeric (if multiple exist)
    text_col = None
    for c in message_cols:
        if df[c].dtype == 'object':
            text_col = c
            break

    if not text_col:
        raise KeyError("Could not locate text-based 'Message' column.")

    # Combine Subject + Message text into one unified text field
    df['text'] = (
        df[subject_col].fillna('') + ' ' + df[text_col].fillna('')
    ).str.strip()

    # Convert spam/ham to numeric labels
    df['spam'] = (
        df[label_col]
        .astype(str)
        .str.lower()
        .map({'spam': 1, 'ham': 0})
        .astype('Int64')
    )

    # Drop unused columns
    df = df[['text', 'spam']].dropna(subset=['text', 'spam'])
    df = df[df['text'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    path = "dataset/enron_spam_data.csv"  # adjust if needed
    df = preprocess_enron_dataset(path)
    print(df.head())
    print(f"✅ Processed {len(df)} rows into standardized ['text', 'spam'] format.")


def load_combined_email_data(base_dir):
    """
    Loads, cleans, and combines all spam-related datasets into one standardized DataFrame.

    Datasets included:
        - emails.csv
        - train.csv
        - email_spam.csv
        - enron_spam_data.csv

    Returns:
        pandas.DataFrame: Combined and preprocessed email data (columns: ['text', 'spam'])
    """

    # Dataset paths
    dataset_path = os.path.join(base_dir, "dataset", "emails.csv")
    train_dataset_path = os.path.join(base_dir, "dataset", "train.csv")
    email_spam_dataset_path = os.path.join(base_dir, "dataset", "email_spam.csv")
    enron_dataset_path = os.path.join(base_dir, "dataset", "enron_spam_data.csv")

    # Sanity check
    print(f"📦 Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load and preprocess all datasets
    datasets = []

    try:
        datasets.append(load_data(dataset_path))
        print(f"✅ Loaded {dataset_path}")
    except Exception as e:
        print(f"⚠️ Skipped emails.csv: {e}")

    try:
        datasets.append(preprocess_train_dataset(train_dataset_path))
        print(f"✅ Loaded train.csv")
    except Exception as e:
        print(f"⚠️ Skipped train.csv: {e}")

    try:
        datasets.append(preprocess_email_spam_dataset(email_spam_dataset_path))
        print(f"✅ Loaded email_spam.csv")
    except Exception as e:
        print(f"⚠️ Skipped email_spam.csv: {e}")

    try:
        datasets.append(preprocess_enron_dataset(enron_dataset_path))
        print(f"✅ Loaded enron_spam_data.csv")
    except Exception as e:
        print(f"⚠️ Skipped enron dataset: {e}")

    # Combine datasets
    if not datasets:
        raise RuntimeError("No datasets were successfully loaded.")

    email_df = pd.concat(datasets, ignore_index=True)
    print(f"🧩 Combined total rows: {len(email_df)}")

    # Clean and preprocess
    data_clean_up(email_df)
    data_preprocessing(email_df)

    # Drop missing or empty text entries
    email_df = email_df.dropna(subset=['text'])
    email_df = email_df[email_df['text'].str.strip() != '']
    email_df.reset_index(drop=True, inplace=True)

    print(f"🧹 Cleaned dataset rows: {len(email_df)}")
    return email_df
