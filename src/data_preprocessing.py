# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Function to load data from an Excel file
def load_data(file_path):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)
    return df

# Function to preprocess data
def preprocess_data(df):
    # Use LabelEncoder to encode the 'generated' column (0: human-written, 1: AI-generated)
    label_encoder = LabelEncoder()
    df['generated'] = label_encoder.fit_transform(df['generated'])
    return df

# Function to split the data into training and testing sets
def split_data(df):
    # Split the data into features (X) and target variable (y)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['generated'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to preprocess text data using TF-IDF vectorization
def preprocess_text_data(X_train, X_test):
    # Fill NaN values in text data with an empty string
    X_train = X_train.fillna('')  
    X_test = X_test.fillna('')

    # Initialize a TF-IDF vectorizer with a maximum of 1000 features and stop words in English
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

    # Transform the training and testing text data using TF-IDF vectorization
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer
