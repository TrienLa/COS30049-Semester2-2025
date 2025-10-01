import pandas as pd
import numpy as np
import plotly.express as pex
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

def generate_graph(email_df):
    """
    Args:
        email_df (panda DataFrame): DataFrame containing information from CSV file.
    """
    spam_counts = email_df['spam'].value_counts()
    fig = pex.bar(spam_counts, x=spam_counts.index, y=spam_counts.values, labels={'x': 'Spam', 'y': 'Count'}, title='Distribution of Spam vs Ham Emails')
    fig.show()

if __name__ == "__main__":
    # 1. Load the email data.
    email_dfs = load_data(sys.path[0] + "/dataset/emails.csv")
    generate_graph(email_dfs)
