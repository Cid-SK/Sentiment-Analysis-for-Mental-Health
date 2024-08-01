import pandas as pd
import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
with open('log_reg.pkl', 'rb') as f:
    logreg_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Data preprocessing functions
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_stopwords(text):
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens  # Return list of tokens

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def process_text(text):
    text = preprocess_text(text)
    tokens = remove_stopwords(text)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

#df = pd.read_csv("G:/SK/Ds/nlp_senti/nlp.csv")

suggestions = {
    'Normal': 'Keep up with your routine and stay healthy!',
    'Depression': 'Consider talking to a mental health professional for support.',
    'Suicidal': 'Please seek immediate help from a mental health crisis service.',
    'Anxiety': 'Practice relaxation techniques and consider talking to a therapist.',
    'Stress': 'Try stress management techniques and ensure you are taking breaks.',
    'Bipolar': 'Regular consultation with a mental health professional is recommended.',
    'Personality Disorder': 'Seek help from a mental health professional for a proper diagnosis and treatment plan.'
}

# Streamlit part
st.set_page_config(page_title="Sentiment Analysis for Mental Health - NLP", layout="wide")
st.title("***:blue[Sentiment Analysis for Mental Health]***")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["***Home***", "***Sentiment Analysis***"])

with tab1:
    st.header("Welcome to the Mental Health Analysis App")
    
    # Introduction
    st.write(
        "This application leverages a comprehensive dataset of mental health statements to provide insights "
        "into various mental health statuses. Use this app to explore the data, view visualizations, and "
        "analyze sentiments."
    )

    st.write(
        "### What You Can Do Here:"
        "\n- **Explore Data:** Get a detailed overview of the dataset."
        "\n- **View Visualizations:** See interactive charts and graphs of mental health data."
        "\n- **Sentiment Analysis:** Enter a statement to receive a mental health category and medical suggestion."
    )

    # Add interactive content
    st.subheader("Interactive Features")
    st.write(
        "- **Visualize Data:** View the distribution of different mental health statuses and token lengths."
        "\n- **Analyze Statements:** Enter a text statement to categorize and get personalized suggestions."
    )

    # Add a quick demo section
    st.subheader("Quick Demo")
    st.write("To see how the sentiment analysis works, try entering a sample statement in the 'Sentiment Analysis' section.")

    # Add links to further resources
    st.subheader("Further Reading")
    st.markdown(
        "[Explore the dataset on Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)"
    )
    st.markdown(
        "[Learn more about mental health](https://www.medicalnewstoday.com/articles/154543#types-of-disorders)"
    )
    
with tab2:
    if 'predict' not in st.session_state:
        st.session_state.predict = False

    c1, c2 = st.columns(2)
    
    with c1:
        user_input = st.text_area("Enter How You Feel!")
        if st.button('Predict'):
           st.session_state.predict = True
    with c2:
        if st.session_state.predict:
            if user_input:
                processed_text = process_text(user_input)
                vectorized_text = loaded_vectorizer.transform([processed_text])
                
                # Predicting the mental health status
                predictions = logreg_model.predict(vectorized_text)

                # Labels
                labels = ['Normal', 'Depression', 'Suicidal', 'Anxiety', 'Stress', 'Bipolar', 'Personality Disorder']

                # Check if the prediction result is as expected
                if len(predictions) == 1 and predictions[0] in labels:
                    label = predictions[0]
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write(f"Predicted Mental Health Status: :blue[{label}]")
                    st.write(f"Suggestion: {suggestions[label]}")
                else:
                    st.write("Unexpected prediction format or label.")
            else:
                st.write("Please enter a statement to get a prediction.")
            st.session_state.predict = False

