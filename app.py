import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Streamlit app starts here

# File uploader for the text and image file
st.title("Word Cloud Generator")

# File uploader for text file
text_file = st.file_uploader("Upload a text file", type=["txt"])
if text_file is not None:
    # Read the uploaded text file
    data = text_file.read().decode("utf-8")

    # Tokenize the sentences and words
    sentences = sent_tokenize(data)
    words = word_tokenize(data)

    # Remove punctuation and keep only alphabetic words
    words = [word.lower() for word in words if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Stem and Lemmatize the words
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Join words back into a processed text
    processed_text = ' '.join(lemmatized_words)

    # Tokenize processed text and get word frequency
    words = nltk.word_tokenize(processed_text)
    word_freq = nltk.FreqDist(words)
    top_words = word_freq.most_common(40)

    # Generate Word Cloud (default)
    wordcloud = WordCloud(width=1000, height=500, random_state=1, background_color='#0A1B27',
                           colormap='Pastel1', collocations=False, stopwords=STOPWORDS).generate_from_frequencies(dict(top_words))

    # Display the WordCloud
    st.subheader("Default WordCloud")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # File uploader for image mask
    image_file = st.file_uploader("Upload an image for word cloud mask", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        # Open and process the uploaded image
        car_mask = np.array(Image.open(image_file))

        # Invert the colors to make white transparent
        car_mask = 255 - car_mask

        # Create WordCloud with car mask
        imgggg = WordCloud(width=1000, height=500, random_state=1, mask=car_mask, background_color='#0d0d0d',
                           colormap='Pastel1', collocations=False, stopwords=STOPWORDS).generate_from_frequencies(dict(top_words))

        # Display the WordCloud with mask
        st.subheader("WordCloud with Mask")
        plt.figure(figsize=(10, 5))
        plt.imshow(imgggg, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
