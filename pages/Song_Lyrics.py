import streamlit as st
import nltk
import string
import pandas as pd
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words, wordnet

# Title of the app
st.title("Lyric Upload and Line Extraction")

# Instructions
st.write("Upload a `.txt` file containing song lyrics or paste lyrics below.")

# File uploader
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

# Text area for pasting lyrics
lyrics_input = st.text_area("Or paste the lyrics here:")

# Function to process lyrics
def process_lyrics(lyrics):
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    df = pd.DataFrame(lines, columns=["Lyric Lines"])
    return df

# Initialize DataFrame
lyrics_df = None

# Handle uploaded file
if uploaded_file:
    lyrics_content = uploaded_file.read().decode("utf-8")
    lyrics_df = process_lyrics(lyrics_content)

# Handle pasted lyrics
if lyrics_input:
    lyrics_df = process_lyrics(lyrics_input)

# Display DataFrame if lyrics are processed
if lyrics_df is not None:
    st.subheader("Extracted Lyric Lines")
    st.dataframe(lyrics_df)
    
    # Option to download DataFrame as CSV
    csv = lyrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="lyrics.csv",
        mime="text/csv",
    )



# Download the necessary NLTK data packages
nltk.download('punkt_tab') # Download the Punkt Tokenizer Models
nltk.download('words')

# Preprocess Data
tokens = word_tokenize(" ".join(df['lyrics'].astype(str))) # Convert the column to string type
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


st.subheader("Generate Word Cloud")
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

st.subheader("Generate Word Cloud: Pie Chart")
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

st.subheader("Generate Word Cloud: Sentiment Type")
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
