import pandas as pd
import numpy as np
import sys

def load_data(filename):
    """
    Args:
        filename (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing emails data,
                          or None if an error occurs.
    """
    try:
        # Read the CSV file and return
        df = pd.read_csv(filename)

        return df
    except Exception as e:
        # Handle errors.
        print(f"Error loading data: {e}")
        return None
    
def data_clean_up(df):
    """
    Args:
        df (DataFrame): DataFrame loaded from the CSV file.
    """
    # Check if there is duplicate entry
    if df.duplicated().sum() > 0:
        print("Found " + str(df.duplicated().sum()) + " duplicate entries")
        df.drop_duplicates(inplace=True) # Remove duplicates from dataframe
        df.reset_index(inplace=True) # Reset the index column after removing duplicates
    return df
    
def data_preprocessing(df):
    """
    Args:
        df (DataFrame): DataFrame loaded from the CSV file.
    """
    df['text'] = df['text'].str.lower() # Lowercase all the text characters
    df['text'] = df['text'].str.replace(r'^subject\s*','',case=False, regex=True) # Remove all the starting subject text
    df['text'] = df['text'].str.replace('\n','',case=False) # Remove all new line

    # Check if there is any null entries
    df = df[df['text'].notnull()]

    # Check if there is invalid entries
    df = df.drop(df.query("`spam` != [0,1]").index)

    # Index reset
    df.reset_index(inplace=True) # Reset the index column after removing duplicates
    df = df.drop('index', axis=1) # Remove the new index column

    # Remove NaN entries for both text and spam
    df = df[['text', 'spam']].dropna()

    # Remove rows with value of float or int type
    df = df.loc[df['text'].apply(type) == str]

    # Convert any invalid entries to string and int for easier checking
    df['text'] = df["text"].apply(lambda x: np.str_(x))
    df['spam'] = df['spam'].astype(int)
    return df

if __name__ == "__main__":
    # Load the email data
    email_dfs = load_data(sys.path[0] + "/processed/clean_dataset/combined_dataset.csv")
    email_dfs = data_preprocessing(data_clean_up(email_dfs))
    print(email_dfs)
