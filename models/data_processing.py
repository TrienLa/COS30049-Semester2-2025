import pandas as pd
import numpy as np
import sys

def load_file(filename):
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
    
def dataframe_normalise(df, method=1):
    """
    Args:
        df (DataFrame): DataFrame loaded from the CSV file.
        method (Int): The clean up method for a specific dataframe structure
    """
    match method:
        case 1: # email_spam.csv structure
            newdf = pd.DataFrame()
            newdf["text"] = df["title"] + ' ' + df["text"]
            newdf["spam"] = (df["type"] == 'spam').astype(int)
            return newdf
        case 2: # train.csv structure
            newdf = pd.DataFrame()
            newdf["text"] = df["text"]
            newdf["spam"] = df["label_num"]
            return newdf
        case 3: # spam_emils_5k5.csv structure
            newdf = pd.DataFrame()
            newdf["text"] = df["Message"]
            newdf["spam"] = (df["Category"] == 'spam').astype(int)
            return newdf
        case _:
            print(f"Invalid method {method}")

def dataframe_combine(df_list):
    """
    Args:
        df_list (list of DataFrame): DataFrame loaded from the CSV file. ex. [df1, df2, df3]
    """
    result_df = pd.concat(df_list) # Concatenate the available list of DataFrame, it should have the same column index after clean up (text, spam)
    return result_df

def dataset_features_extract(df, feature=None):
    """
    Args:
        df (DataFrame): DataFrame loaded from the CSV file.
        feature (String) : [length | frequency | presence] The feature you want to extract from the dataset
    """
    match feature:
        case "length":
            df["text_length"] = (df["text"].apply(lambda x: np.str_(x))).apply(len)
            #df.to_csv(sys.path[0] + "/dataset/extracted/length.csv",index=False)
            return
        case "frequency":
            df["$_frequency"] = df['text'].apply(lambda x: x.count('$'))
            return
        case "presence":
            df["money_presence"] = (df["text"].str.contains('money')).astype(int)
            return
        case _:
            print("No matching feature")


if __name__ == "__main__":
    # Load the files to dataframe
    dataframe_list = []
    dataframe_list.append(load_file(sys.path[0] + "/dataset/emails.csv"))
    dataframe_list.append(load_file(sys.path[0] + "/dataset/email_spam.csv"))
    dataframe_list.append(load_file(sys.path[0] + "/dataset/train.csv"))
    dataframe_list.append(load_file(sys.path[0] + "/dataset/spam_emails_5k5.csv"))

    # Normalise the dataframes
    dataframe_list[1] = dataframe_normalise(dataframe_list[1], 1)
    dataframe_list[2] = dataframe_normalise(dataframe_list[2], 2)
    dataframe_list[3] = dataframe_normalise(dataframe_list[3], 3)

    # Combine all the cleaned up dataframes
    combined_df = dataframe_combine(dataframe_list)

    # Extract dataset features
    dataset_features_extract(combined_df, "length")
    dataset_features_extract(combined_df, "frequency")
    dataset_features_extract(combined_df, "presence")

    # Output the dataframe to csv
    combined_df.to_csv(sys.path[0] + "/processed/clean_dataset/combined_dataset.csv", index=False)
