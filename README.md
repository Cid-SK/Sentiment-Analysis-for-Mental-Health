---

# Sentiment Prediction for Mental Health-NLP

## Overview

This application leverages a sentiment analysis model to categorize mental health statements and provide tailored suggestions based on textual input. The model is trained using a comprehensive dataset to identify various mental health statuses and offer appropriate recommendations.

## Features

- **Sentiment Analysis:** Input a text statement to categorize it and receive a personalized suggestion based on the predicted mental health status.
- **Home Page:** Introduction to the application, features, and further reading resources.

## Requirements

To run this application, you need to install the following Python packages. You can install them using pip:

```bash
pip install pandas streamlit pickle nltk scikit-learn
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd <project-directory>
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Your Environment:**
   Ensure you have the required packages installed. Use the `requirements.txt` file to install dependencies.

2. **Run the Streamlit Application Locally:**
   Navigate to the project directory and run:

   ```bash
   streamlit run app.py
   ```

3. **Interact with the App:**
   - **Home Tab:** Provides an introduction, features, and further reading resources.
   - **Sentiment Analysis Tab:** Allows users to input text, predict mental health status, and provide suggestions.

## Deployment

The application is deployed and accessible via the following link:

- [Sentiment Analysis for Mental Health - Live App](https://sentiment-analysis-for-mental-health-1.onrender.com/)

## Code Overview

- **Data Preprocessing:** Text data is preprocessed by converting to lowercase, removing unwanted characters, tokenizing, removing stopwords, and lemmatizing.
- **Model and Vectorizer Loading:** The Logistic Regression model and TF-IDF Vectorizer are loaded using pickle.
- **Streamlit Interface:**
  - **Home Tab:** Displays an introduction and links to further resources.
  - **Sentiment Analysis Tab:** Allows users to input text, predict mental health status, and provide suggestions.

## Links

- [Explore the dataset on Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)
- [Learn more about mental health](https://www.medicalnewstoday.com/articles/154543#types-of-disorders)

---
