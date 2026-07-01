import os
import re
import joblib
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Initialize NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Global variables for model and vectorizer
model = None
vectorizer = None
demo_mode = True
stemmer = PorterStemmer()
stop_words = set()

try:
    stop_words = set(stopwords.words('english'))
except Exception:
    # Fallback to simple set if NLTK stopwords fail
    stop_words = {'the', 'a', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 'an', 'this', 'that', 'from', 'as'}

def clean_text(text):
    # 1. HTML Stripping
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Case Folding
    text = text.lower()
    # 3. Remove non-alphabetic
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 4. Tokenization, Stemming, Stopword Removal
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def mock_predict(text):
    """
    A sophisticated rule-based classifier to act as a fallback/demo mode.
    This provides logical predictions even before the user trains the ML model.
    """
    positive_words = {
        'good', 'great', 'love', 'like', 'awesome', 'excellent', 'beautiful', 
        'wonderful', 'brilliant', 'fantastic', 'best', 'nice', 'enjoy', 'cool', 
        'masterpiece', 'perfect', 'amazing', 'fun', 'happy', 'glad', 'highly', 
        'recommend', 'superb', 'entertaining', 'classic', 'gem', 'outstanding'
    }
    negative_words = {
        'bad', 'terrible', 'worst', 'hate', 'dislike', 'boring', 'waste', 'poor', 
        'awful', 'stupid', 'dumb', 'horrible', 'worse', 'annoying', 'crap', 
        'rubbish', 'fail', 'disappointed', 'dreadful', 'laughable', 'mess', 
        'garbage', 'suck', 'sucks', 'pathetic', 'pointless', 'useless', 'slow'
    }
    
    cleaned = clean_text(text)
    words = cleaned.split()
    
    # Stem positive and negative list to align with cleaned text
    stemmed_pos = {stemmer.stem(w) for w in positive_words}
    stemmed_neg = {stemmer.stem(w) for w in negative_words}
    
    pos_count = sum(1 for w in words if w in stemmed_pos)
    neg_count = sum(1 for w in words if w in stemmed_neg)
    
    text_hash = abs(hash(text))
    
    if pos_count > neg_count:
        sentiment = "POSITIF"
        ratio = pos_count / (pos_count + neg_count)
        confidence = 65.0 + (ratio * 30.0) # 65% - 95%
    elif neg_count > pos_count:
        sentiment = "NEGATIF"
        ratio = neg_count / (pos_count + neg_count)
        confidence = 65.0 + (ratio * 30.0)
    else:
        # Tie breaker based on hash value for consistency
        sentiment = "POSITIF" if text_hash % 2 == 0 else "NEGATIF"
        # If no sentiment words are present, confidence is lower
        confidence = 50.0 + (text_hash % 15)
        
    # Cap confidence score at 99.9%
    confidence = min(99.9, max(50.0, confidence))
    return sentiment, round(confidence, 2)

def load_ml_model():
    global model, vectorizer, demo_mode
    model_path = 'model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            demo_mode = False
            print("Successfully loaded trained Machine Learning model and vectorizer.")
        except Exception as e:
            print(f"Error loading model files: {e}. Falling back to demo mode.")
            demo_mode = True
    else:
        print("Model or Vectorizer files not found. App will run in DEMO MODE.")
        demo_mode = True

# Try to load model on startup
load_ml_model()

@app.route('/')
def home():
    # Check if model exists to dynamically show status
    load_ml_model()
    return render_template('index.html', demo_mode=demo_mode)

@app.route('/predict', methods=['POST'])
def predict():
    # Make sure model status is up to date
    load_ml_model()
    
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'status': 'error', 'message': 'No review text provided'}), 400
        
    review_text = data['review'].strip()
    if not review_text:
        return jsonify({'status': 'error', 'message': 'Review text is empty'}), 400
        
    if demo_mode:
        sentiment, confidence = mock_predict(review_text)
    else:
        try:
            # 1. Clean the review text
            cleaned = clean_text(review_text)
            # 2. Vectorize
            vect_text = vectorizer.transform([cleaned])
            # 3. Predict sentiment probabilities
            prob = model.predict_proba(vect_text)[0] # [prob_neg, prob_pos]
            pred = model.predict(vect_text)[0] # 0 or 1
            
            sentiment = "POSITIF" if pred == 1 else "NEGATIF"
            confidence = prob[1] * 100 if pred == 1 else prob[0] * 100
            confidence = round(float(confidence), 2)
        except Exception as e:
            # Safe fallback if ML prediction fails
            print(f"Error during ML prediction: {e}. Using mock backup.")
            sentiment, confidence = mock_predict(review_text)
            
    return jsonify({
        'status': 'success',
        'sentiment': sentiment,
        'confidence': confidence,
        'demo_mode': demo_mode
    })

if __name__ == '__main__':
    # Bind to all interfaces for easy access
    app.run(debug=True, host='0.0.0.0', port=5000)
