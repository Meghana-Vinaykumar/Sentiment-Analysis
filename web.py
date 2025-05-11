#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}

model = load_model("imdb.h5")

max_len = 200

def predict_sentiment(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    sequence = [1]  
    for word in tokens:
        sequence.append(word_index.get(word, 2)) 
    padded = pad_sequences([sequence], maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment, prediction

st.title("Sentiment Analyzer")
st.write("Enter a review below to predict its sentiment.")

user_input = st.text_area("Your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.title("Prediction")
        st.write(f"**Sentiment:** {sentiment}")

