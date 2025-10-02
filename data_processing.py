import pandas as pd
import numpy as np
import plotly.express as pex
import plotly.io
import os, sys, re

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
        df (dataframe): Dataframe loaded from the CSV file.
    """
    # Check if there is duplicate entry
    if df.duplicated().sum() > 0:
        print("Found " + str(df.duplicated().sum()) + " duplicate entries")
        df.drop_duplicates(inplace=True) # Remove duplicates from dataframe
        df.reset_index(inplace=True) # Reset the index column after removing duplicates
    print(df)
    
def data_preprocessing(df):
    """
    Args:
        df (dataframe): Dataframe loaded from the CSV file.
    """
    df['text'] = df['text'].str.lower() # Lowercase all the text characters
    df['text'] = df['text'].str.replace(r'^subject\s*','',case=False, regex=True) # Remove all the starting subject text


def generate_feature_plot(email_df):
    """
    Args:
        email_df (pandas DataFrame): DataFrame containing information from CSV file.
    """
    # Generate plot for distribution of spam vs ham in the dataset 
    spam_counts = email_df['spam'].value_counts()
    spam_distribution = pex.bar(spam_counts, x=spam_counts.index, y=spam_counts.values, labels={'x': 'Spam', 'y': 'Count'}, title='Distribution of Spam vs Ham emails')
    plotly.io.write_image(spam_distribution, sys.path[0] + "/plots/spam_distribution.png")

    # Generate plot for text length distribution of spam vs ham in the dataset 
    email_df['text length'] = email_df['text'].apply(len) # Add a new data column categorizing the length of each email
    email_length = pex.histogram(email_df, y='text length', color='spam', nbins=35, title='Text length distribution by Spam vs Ham')
    plotly.io.write_image(email_length, sys.path[0] + "/plots/email_length.png")

    # Generate a word cloud for non spam emails

    # Generate a word cloud for spam emails

    # Generate a histogram of keyword frequencies

    

if __name__ == "__main__":
    # Load the email data
    email_dfs = load_data(sys.path[0] + "/dataset/emails.csv")
    data_clean_up(email_dfs)
    data_preprocessing(email_dfs)
    # Generate plot
    # generate_feature_plot(email_dfs)
