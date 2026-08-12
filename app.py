import streamlit as st
import pickle
import re
import string

# Load trained model
with open("models/lr_model.pkl", "rb") as f:
    LR = pickle.load(f)

# Load TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorization = pickle.load(f)


# Text preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    return text


# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detection")

st.write(
    "Enter a news article below to check whether it is Fake News or Not Fake News."
)

# News input
news = st.text_area(
    "Enter News Article",
    height=120,
    placeholder="Paste the news article here..."
)

# Prediction
if st.button("🔍 Predict News"):

    if news.strip() == "":
        st.warning("Please enter some news text.")

    else:
        cleaned_news = wordopt(news)

        news_vector = vectorization.transform([cleaned_news])

        prediction = LR.predict(news_vector)[0]

        if prediction == 0:
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ NOT A FAKE NEWS")