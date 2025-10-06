import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from data_processing import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import pickle
import sys

def generate_feature_plot(email_df):
    """
    Args:
        email_df (pandas DataFrame): DataFrame containing information from CSV file.
    """
    # Generate plot for distribution of spam vs ham in the dataset 
    spam_distribution = sns.displot(email_df, x='spam', discrete=True, shrink=0.8).set_xlabels('Spam').set(title='Distribution of Spam vs Ham emails')
    spam_distribution.savefig(sys.path[0] + "processed/plots/spam_distribution.png")

    # Generate plot for text length distribution of spam vs ham in the dataset 
    email_df['text length'] = email_df['text'].apply(len) # Add a new data column categorizing the length of each email
    email_length = sns.displot(email_df, x='text length', multiple="stack", hue="spam", bins=35).set_xlabels('Text Length').set(title='Text length distribution by Spam vs Ham emails')
    email_length.savefig(sys.path[0] + "processed/plots/email_length.png")

    # Generate a word cloud for non spam emails
    ham_text = " ".join(email for email in email_df[email_df['spam'] == 0]['text']) # Add all the text in the dataset that isn't spam to a long string
    ham_wordcloud = WordCloud(width=500, height=500, max_font_size=75, max_words=150, background_color="white").generate(ham_text)
    ham_wordcloud.to_file(sys.path[0] + "processed/plots/ham_wordcloud.png")

    # Generate a word cloud for spam emails
    spam_text = " ".join(email for email in email_df[email_df['spam'] == 1]['text'])
    spam_wordcloud = WordCloud(width=500, height=500, max_font_size=75, max_words=150, background_color="white").generate(spam_text)
    spam_wordcloud.to_file(sys.path[0] + "processed/plots/spam_wordcloud.png")

    # Generate a histogram of keyword frequencies

def conf_matrix(model, x_test, y_test):
    """
    Args:
        model (str): String of the model file that we will load up.

        X_test (DataFrame): X_test data split from the vectorized DataFrame
    """
    # Load data from pickle file
    with open(f'processed/models/{model}.pkl', 'rb') as tm:
        new_pipe = pickle.load(tm)
        
    # Prediction data
    y_predict = new_pipe.predict(x_test)
    # Calculate confusion matrix from predicted data
    model_conf_matrix = confusion_matrix(y_predict, y_test)

    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    model_conf_matrix.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    # Generate confusion matrix plot
    cmatrix_axes = sns.heatmap(model_conf_matrix, annot=labels, fmt='', cmap='Blues')
    cmatrix_axes.set(xlabel="Predicted Label", ylabel="True Label")
    cmatrix_axes.set(title=f'Confusion Matrix of {model}')
    cmatrix_plot = cmatrix_axes.get_figure()
    cmatrix_plot.savefig(f"processed/confusion_matrix/{model}.png")

if __name__ == "__main__":
    # Load the email data
    email_dfs = load_data(sys.path[0] + "/dataset/emails.csv")
    # Generate plot
    generate_feature_plot(email_dfs)