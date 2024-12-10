# Import Libraries
import requests
import os
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import subprocess
import sys
import nltk
import string
from nltk.corpus import stopwords, words, wordnet
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex

wordnet.synsets('na')

# URL of the font file
# Added 'https://' to the URL
font_url = "https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf"

# File path to save the font file
font_path = "thsarabunnew-webfont.ttf"

# Download the font file
response = requests.get(font_url)
with open(font_path, 'wb') as f:
    f.write(response.content)

text_path = "Beyonce.csv"
# Specify encoding explicitly
df = pd.read_csv(text_path, encoding='latin1')

df.head()


# Download the necessary NLTK data packages
nltk.download('punkt_tab') # Download the Punkt Tokenizer Models
nltk.download('words')

english_words = set(words.words())

# Preprocess Data
tokens = word_tokenize(" ".join(df['Lyric']))
english_stopwords = set(stopwords.words('english'))

# Filter out stopwords, punctuation, and words starting with apostrophe
filtered_tokens = [word.lower() for word in tokens
                   if word.strip()
                   and word.lower() not in english_stopwords
                   and wordnet.synsets(word)
                   and len(word) > 2
                   and word.lower() in english_words]

# Join tokens into a single string
text_for_wordcloud = ' '.join(filtered_tokens)

st.write(len(filtered_tokens))
st.write(filtered_tokens[:100])

# Create a pandas Series from the list 📑
word_series = pd.Series(filtered_tokens)
word_counts = word_series.value_counts().reset_index()

# Rename the columns for clarity
word_counts.columns = ["Word", "Count"]

# Display the word counts table
word_counts.head(20).style.background_gradient(cmap='YlGn')

# Define colors using a colormap
colors = plt.cm.YlGn(np.linspace(0.8, 0.1, len(word_counts["Word"][:15])))

# Plot the top 15 most frequent words with gradient color
plt.figure(figsize=(12, 8))
bars = plt.bar(word_counts["Word"][:15], word_counts["Count"][:15], color=colors)

# Add color gradient to bars
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.xlabel("Words", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Top Most Frequent Words", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(plt.gcf())

# Create a WordCloud using the downloaded font
wordcloud = WordCloud(
                      relative_scaling=0.3,
                      min_font_size=1,
                      background_color = "white",
                      width=1024,
                      height=768,
                      max_words=200,
                      colormap='plasma', 
                      scale=3,
                      font_step=4,
                      collocations=False,
                      margin=5
                      ).generate(' '.join(filtered_tokens))

# Generate the WordCloud
wordcloud.generate(text_for_wordcloud)

# Display Word Cloud 🛫
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt.gcf())

# Initialize the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Calculate the sentiment scores for the entire text
sentiment_score = analyzer.polarity_scores(text_for_wordcloud)

# Extract the individual scores
neg = sentiment_score['neg']
neu = sentiment_score['neu']
pos = sentiment_score['pos']

sentiment_score

# Create labels for the pie chart
labels = ['Negative', 'Neutral', 'Positive']

# Create sizes for each sentiment category
sizes = [neg, neu, pos]

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=plt.cm.Set2.colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')
st.pyplot(plt.gcf())

# Initialize list to store word sentiments
word_sentiments = []

# Initialize variables to accumulate sentiment scores
positive_score_sum = 0
negative_score_sum = 0
neutral_score_sum = 0
positive_count = 0
negative_count = 0
neutral_count = 0
total_words = 0
# Count occurrences of each sentiment and accumulate sentiment scores
for word in filtered_tokens:
    total_words +=1
    sentiment_score = analyzer.polarity_scores(word)['compound']
    if sentiment_score >= 0.05:
        word_sentiments.append(('Positive', word))
        positive_count += 1
        positive_score_sum += sentiment_score
    elif sentiment_score <= (-0.05):
        word_sentiments.append(('Negative', word))
        negative_count += 1
        negative_score_sum += sentiment_score
    else:
        word_sentiments.append(('Neutral', word))
        neutral_count += 1
        neutral_score_sum += sentiment_score

# Calculate average sentiment score for each emotion
average_positive_score = positive_score_sum / positive_count if positive_count != 0 else 0
average_negative_score = negative_score_sum / negative_count if negative_count != 0 else 0
average_neutral_score = neutral_score_sum / neutral_count if neutral_count != 0 else 0
average_total_score = (positive_score_sum + negative_score_sum + neutral_score_sum) / total_words

# Print results
st.write("pos:", positive_count)
st.write("neg:", negative_count)
st.write("neu:", neutral_count)
st.write("Total:", total_words)

# Print results with two decimal places
st.write("Average pos Score: {:.2f}".format(average_positive_score))
st.write("Average neg Score: {:.2f}".format(average_negative_score))
st.write("Average neu Score: {:.2f}".format(average_neutral_score))
st.write("Average Total Score: {:.2f}".format(average_total_score))

# Create separate lists for each word type
all_words = [word for sentiment, word in word_sentiments]
positive_words = [word for sentiment, word in word_sentiments if sentiment == 'Positive']
negative_words = [word for sentiment, word in word_sentiments if sentiment == 'Negative']
neutral_words = [word for sentiment, word in word_sentiments if sentiment == 'Neutral']

# Count occurrences of each word type
word_counts = pd.Series(all_words).value_counts().head(30)
positive_word_counts = pd.Series(positive_words).value_counts().head(30)
negative_word_counts = pd.Series(negative_words).value_counts().head(30)
neutral_word_counts = pd.Series(neutral_words).value_counts().head(30)

# Create a DataFrame to display the top of each word type
data = {
    'All Words': word_counts.index,
    'All Counts': word_counts.values,
    'Pos Words': positive_word_counts.index,
    'Pos Counts': positive_word_counts.values,
    'Neg Words': negative_word_counts.index,
    'Neg Counts': negative_word_counts.values,
    'Neu Words': neutral_word_counts.index,
    'Neu Counts': neutral_word_counts.values
}

df = pd.DataFrame(data)

# Set column names
df.columns = ["Word", "Count", "Pos Words", "Pos Counts", "Neg Words", "Neg Counts", "Neu Words", "Neu Counts"]

# Display the DataFrame with styled background gradients
styled_df = df.head(30).style.background_gradient(cmap='YlGn')
st.write(styled_df)

# Generate word clouds for each sentiment type
wordclouds = {}
for sentiment_type in ['Positive', 'Negative', 'Neutral']:
    words = [word[1] for word in word_sentiments if word[0] == sentiment_type]
    if words:
        wordcloud = WordCloud(relative_scaling=0.3,
                              min_font_size=1,
                              background_color="white",
                              width=1024,
                              height=768,
                              max_words=500,
                              colormap='plasma',
                              scale=3,
                              font_step=4,
                              collocations=False,
                              margin=5).generate(' '.join(words))
        wordclouds[sentiment_type] = wordcloud

# Display the word clouds  
plt.figure(figsize=(18, 6))
for i, sentiment_type in enumerate(['Positive', 'Negative', 'Neutral']):
    if sentiment_type in wordclouds:
        plt.subplot(1, 3, i+1)
        plt.imshow(wordclouds[sentiment_type], interpolation='bilinear')
        plt.title(sentiment_type + ' Words')
        plt.axis('off')
st.pyplot(plt.gcf())





text_object = NRCLex(' '.join(filtered_tokens))
text_object.affect_frequencies

text_object.top_emotions

# Get the total number of words
total_words = len(text_object.words)
st.write("Total words in the text:", total_words)

sentiment_scores = pd.DataFrame(list(text_object.raw_emotion_scores.items()))
sentiment_scores = sentiment_scores.rename(columns={0: "Sentiment", 1: "Count"})
sentiment_scores.style.background_gradient(cmap='YlGn')

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sentiment_scores['Count'], labels=sentiment_scores['Sentiment'], colors=plt.cm.tab20c.colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')
st.pyplot(plt.gcf())

from collections import Counter

# Define color maps for each sentiment
sentiment_cmaps = {
    'fear': 'viridis',
    'anger': 'magma',
    'anticipation': 'viridis',
    'trust': 'cividis',
    'surprise': 'cool',
    'positive': 'viridis',
    'negative': 'inferno',
    'sadness': 'inferno',
    'disgust': 'copper',
    'joy': 'spring'
}

# Get all unique sentiments in the text
unique_sentiments = list(set(sentiment for emotions in text_object.affect_dict.values() for sentiment in emotions))

# Create a pandas Series from the list of filtered tokens
word_series = pd.Series(filtered_tokens)

# Create a dictionary to store word frequencies for each sentiment
sentiment_word_counts = {}

# Calculate word frequencies for each sentiment
for sentiment in unique_sentiments:
    words_for_sentiment = [word for word, emotions in text_object.affect_dict.items() if sentiment in emotions]
    sentiment_word_counts[sentiment] = word_series[word_series.isin(words_for_sentiment)].value_counts().head(10)

# Plotting the bar graph for each sentiment
num_rows = len(unique_sentiments) // 2 + (1 if len(unique_sentiments) % 2 != 0 else 0)
num_cols = 2
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 6))
axs = axs.ravel()

for i, sentiment in enumerate(unique_sentiments):
    word_counts = sentiment_word_counts[sentiment]
    axs[i].barh(word_counts.index, word_counts.values, color=plt.cm.YlGn(word_counts.values / max(word_counts.values)), alpha=0.9)
    axs[i].set_title(f'Top 10 Words for {sentiment.capitalize()} Sentiment')
    axs[i].set_xlabel('Frequency')
    axs[i].tick_params(axis='y')
    axs[i].set_yticklabels(word_counts.index)
    axs[i].invert_yaxis()

# Hide any remaining empty subplots
for j in range(len(unique_sentiments), num_rows * num_cols):
    axs[j].axis('off')

plt.tight_layout()
st.pyplot(plt.gcf())

# Determine the number of rows and columns needed
num_rows = len(unique_sentiments) // 3 + (1 if len(unique_sentiments) % 3 != 0 else 0)
num_cols = 3

# Create subplots with 3 columns
fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 6))

# Flatten the axis array
axs = np.ravel(axs)

# Calculate word frequencies for each sentiment
for sentiment in unique_sentiments:
    words_for_sentiment = [word for word, emotions in text_object.affect_dict.items() if sentiment in emotions]
    sentiment_word_counts[sentiment] = word_series[word_series.isin(words_for_sentiment)].value_counts()

# Create a WordCloud for each sentiment
for i, sentiment in enumerate(unique_sentiments):
    words_for_sentiment = [word for word, emotions in text_object.affect_dict.items() if sentiment in emotions]
    word_counts = sentiment_word_counts[sentiment]

    # Create a WordCloud
    wordcloud = WordCloud(
        relative_scaling=0.3,
        min_font_size=1,
        background_color="white",
        width=400,
        height=300,
        max_words=500,
        colormap=sentiment_cmaps.get(sentiment),
        scale=3,
        font_step=4,
        collocations=False,
        margin=5
    )

    # Generate WordCloud from word frequencies
    wordcloud.generate_from_frequencies(word_counts.to_dict())

    # Display the WordCloud in the appropriate subplot
    axs[i].imshow(wordcloud, interpolation='bilinear')
    axs[i].set_title(f"Word Cloud for {sentiment.capitalize()} Sentiment")
    axs[i].axis('off')

    # Hide any remaining empty subplots
for j in range(len(unique_sentiments), num_rows * num_cols):
    axs[j].axis('off')

# Adjust layout
plt.tight_layout()
st.pyplot(plt.gcf())
