import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱", layout="centered")

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    refined_tokens = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            refined_tokens.append(stemmer.stem(word))
            
    return " ".join(refined_tokens)

@st.cache_resource
def load_models():
    models = {}
    if not os.path.exists('models/lr_model.pkl'):
        st.error("Model files not found! Please make sure the 'models' directory exists with the .pkl files.")
        st.stop()
        
    with open('models/lr_model.pkl', 'rb') as f:
        models['lr'] = pickle.load(f)
    with open('models/nb_model.pkl', 'rb') as f:
        models['nb'] = pickle.load(f)
    with open('models/svm_model.pkl', 'rb') as f:
        models['svm'] = pickle.load(f)
    with open('models/tfidf.pkl', 'rb') as f:
        models['tfidf'] = pickle.load(f)
    return models

models = load_models()

st.title("📱 SMS Spam Classifier")
st.markdown("Enter the content of an SMS message below to instantly check if it is **Spam** or **Not Spam (Ham)**.")

model_choice = st.selectbox(
    "Select the ML Model to use for Prediction:",
    ["Logistic Regression", "Multinomial Naive Bayes", "Linear Support Vector Classifier"]
)

sms_input = st.text_area("SMS Content", height=200, placeholder="Paste your SMS text here...")

if st.button("Classify SMS", type="primary"):
    if not sms_input.strip():
        st.warning("⚠️ Please enter some text to classify.")
    else:
        with st.spinner("Analyzing text..."):
            transformed_sms = transform_text(sms_input)
            
            vector_input = models['tfidf'].transform([transformed_sms])
            
            if model_choice == "Logistic Regression":
                result = models['lr'].predict(vector_input)[0]
            elif model_choice == "Multinomial Naive Bayes":
                result = models['nb'].predict(vector_input)[0]
            else:
                result = models['svm'].predict(vector_input)[0]
                
            st.divider()
            if result == 1:
                st.error("🚨 **Prediction:** This SMS is classified as **SPAM**.")
            else:
                st.success("✅ **Prediction:** This SMS is classified as **NOT SPAM** (Ham).")
