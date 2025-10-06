import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from data_processing import load_data
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

if __name__ == "__main__":
    # Load the email data
    email_dfs = load_data(sys.path[0] + "/dataset/emails.csv")
    # Generate plot
    generate_feature_plot(email_dfs)