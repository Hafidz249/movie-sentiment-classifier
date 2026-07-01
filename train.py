import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded
print("Initializing NLTK resources...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text, stemmer, stop_words):
    # 1. HTML Stripping
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Case Folding
    text = text.lower()
    
    # 3. Keep only alphabets and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Tokenization & Stopwords & Stemming
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(cleaned_words)

def train_model():
    dataset_path = os.path.join('data', 'IMDB Dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}!")
        print("Please place the Kaggle 'IMDB Dataset.csv' in the 'data' directory.")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    print(f"Dataset shape: {df.shape}")
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    print("Initializing preprocessing tools (Stemmer & Stopwords)...")
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    print("Cleaning review texts (this might take a few minutes)...")
    # Clean all reviews
    df['cleaned_review'] = df['review'].apply(lambda x: clean_text(str(x), stemmer, stop_words))
    
    print("Splitting dataset into train and test sets...")
    X = df['cleaned_review']
    y = df['sentiment'].apply(lambda x: 1 if str(x).lower() == 'positive' else 0) # 1 for positive, 0 for negative
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    print("Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating Model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print("Saving model and vectorizer...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("Model ('model.pkl') and Vectorizer ('vectorizer.pkl') saved successfully!")

if __name__ == '__main__':
    train_model()
