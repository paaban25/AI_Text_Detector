
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    
    df = pd.read_excel(file_path)
    return df


def preprocess_data(df):
   
    label_encoder = LabelEncoder()
    df['generated'] = label_encoder.fit_transform(df['generated'])
    return df

def split_data(df):
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['generated'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
def preprocess_text_data(X_train, X_test):
    X_train = X_train.fillna('')  
    X_test = X_test.fillna('')
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer
