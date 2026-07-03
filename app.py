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

# ─────────────────────────────────────────────
# Bahasa Indonesia Support
# ─────────────────────────────────────────────

# Stopwords Bahasa Indonesia
INDONESIAN_STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'adalah', 'pada', 'tidak', 'ada', 'dalam', 'juga', 'saya', 'anda',
    'kamu', 'mereka', 'kita', 'kami', 'dia', 'ia', 'oleh', 'atau', 'tapi',
    'tetapi', 'namun', 'karena', 'sebagai', 'sudah', 'telah', 'akan', 'bisa',
    'dapat', 'hal', 'saja', 'hanya', 'jika', 'kalau', 'lebih', 'masih',
    'belum', 'sudah', 'bahwa', 'agar', 'supaya', 'ketika', 'setelah',
    'sebelum', 'selama', 'antara', 'seperti', 'maka', 'pun', 'lagi',
    'paling', 'sangat', 'sekali', 'banyak', 'semua', 'setiap', 'masing',
    'satu', 'dua', 'nya', 'mu', 'ku', 'kah', 'lah', 'pun'
}

# Kamus terjemahan Indonesia → Inggris (domain ulasan film)
ID_TO_EN_DICT = {
    # Kata sentimen positif
    'bagus': 'good', 'baik': 'good', 'hebat': 'great', 'luar': 'great',
    'biasa': 'ordinary', 'keren': 'cool', 'kece': 'cool', 'mantap': 'awesome',
    'suka': 'like', 'cinta': 'love', 'indah': 'beautiful', 'cantik': 'beautiful',
    'menakjubkan': 'amazing', 'memukau': 'amazing', 'spektakuler': 'spectacular',
    'brilian': 'brilliant', 'sempurna': 'perfect', 'terbaik': 'best',
    'menarik': 'interesting', 'menghibur': 'entertaining', 'seru': 'fun',
    'lucu': 'funny', 'menyentuh': 'touching', 'emosional': 'emotional',
    'luar biasa': 'outstanding', 'terpesona': 'captivated', 'memikat': 'captivating',
    'epik': 'epic', 'fantastik': 'fantastic', 'top': 'top', 'oke': 'okay',
    'senang': 'happy', 'puas': 'satisfied', 'memuaskan': 'satisfying',
    'rekomendasi': 'recommend', 'rekomendasikan': 'recommend',
    'wajib': 'must', 'tonton': 'watch', 'nonton': 'watch',
    'mahakarya': 'masterpiece', 'gemilang': 'brilliant', 'mengesankan': 'impressive',
    'lezat': 'delightful', 'menyenangkan': 'enjoyable', 'mengagumkan': 'admirable',
    'unik': 'unique', 'orisinal': 'original', 'inovatif': 'innovative',
    'kreatif': 'creative', 'ciamik': 'great', 'jempol': 'thumbs up',
    'tepat': 'appropriate', 'pas': 'perfect', 'cocok': 'fitting',
    # Kata sentimen negatif
    'buruk': 'bad', 'jelek': 'bad', 'parah': 'terrible', 'sampah': 'garbage',
    'payah': 'pathetic', 'gagal': 'fail', 'kecewa': 'disappointed',
    'mengecewakan': 'disappointing', 'membosankan': 'boring', 'bosan': 'boring',
    'ngantuk': 'boring', 'menjemukan': 'tedious', 'lambat': 'slow',
    'lemah': 'weak', 'hancur': 'ruined', 'rusak': 'broken', 'cacat': 'defective',
    'murahan': 'cheap', 'asal': 'mediocre', 'bikin': 'make', 'mual': 'sick',
    'muak': 'disgusted', 'jijik': 'disgusting', 'busuk': 'rotten',
    'terburuk': 'worst', 'paling jelek': 'worst', 'percuma': 'useless',
    'buang': 'waste', 'sia': 'waste', 'sia-sia': 'waste', 'rugi': 'loss',
    'tidak bagus': 'bad', 'tidak baik': 'bad', 'tidak suka': 'dislike',
    'benci': 'hate', 'membenci': 'hate', 'menyebalkan': 'annoying',
    'menyebal': 'annoying', 'kesal': 'annoyed', 'frustasi': 'frustrated',
    'konyol': 'stupid', 'bodoh': 'stupid', 'tolol': 'dumb', 'gaje': 'random',
    'gak jelas': 'unclear', 'tidak jelas': 'unclear', 'membingungkan': 'confusing',
    'bingung': 'confused', 'absurd': 'absurd', 'aneh': 'weird', 'ganjil': 'strange',
    'norak': 'cheesy', 'klise': 'cliche', 'picisan': 'trashy',
    'dangkal': 'shallow', 'hambar': 'bland', 'flat': 'flat', 'datar': 'flat',
    'prediktabel': 'predictable', 'membuat ngantuk': 'boring',
    # Kata umum domain film
    'film': 'film', 'movie': 'movie', 'cerita': 'story', 'plot': 'plot',
    'akting': 'acting', 'aktor': 'actor', 'aktris': 'actress', 'sutradara': 'director',
    'sinematografi': 'cinematography', 'efek': 'effects', 'visual': 'visual',
    'musik': 'music', 'soundtrack': 'soundtrack', 'dialog': 'dialog',
    'karakter': 'character', 'tokoh': 'character', 'peran': 'role',
    'adegan': 'scene', 'penampilan': 'performance', 'akhir': 'ending',
    'awal': 'beginning', 'tengah': 'middle', 'alur': 'plot', 'twist': 'twist',
    'genre': 'genre', 'horor': 'horror', 'komedi': 'comedy', 'romantis': 'romantic',
    'aksi': 'action', 'petualangan': 'adventure', 'drama': 'drama',
    'thriller': 'thriller', 'animasi': 'animation', 'dokumenter': 'documentary',
    'layar': 'screen', 'bioskop': 'cinema', 'penonton': 'audience',
    'ulasan': 'review', 'rating': 'rating', 'bintang': 'star',
    # Intensifier & modifier
    'sangat': 'very', 'banget': 'very', 'sekali': 'very', 'amat': 'extremely',
    'benar': 'really', 'betul': 'really', 'sungguh': 'truly', 'cukup': 'quite',
    'lumayan': 'fairly', 'agak': 'somewhat', 'sedikit': 'slightly',
    'jauh': 'far', 'lebih': 'more', 'kurang': 'less', 'tidak': 'not',
    'bukan': 'not', 'jangan': 'dont', 'hampir': 'almost', 'nyaris': 'nearly',
    'terlalu': 'too', 'cukup': 'enough',
}

# Kata sentimen Bahasa Indonesia untuk mock_predict
IDN_POSITIVE_WORDS = {
    'bagus', 'baik', 'hebat', 'keren', 'mantap', 'suka', 'cinta', 'indah',
    'cantik', 'menakjubkan', 'memukau', 'spektakuler', 'brilian', 'sempurna',
    'terbaik', 'menarik', 'menghibur', 'seru', 'lucu', 'menyentuh', 'epik',
    'fantastik', 'senang', 'puas', 'memuaskan', 'mahakarya', 'gemilang',
    'mengesankan', 'menyenangkan', 'ciamik', 'kece', 'rekomendasi', 'wajib',
    'luar biasa', 'orisinal', 'kreatif', 'emosional', 'unik', 'inovatif',
}

IDN_NEGATIVE_WORDS = {
    'buruk', 'jelek', 'parah', 'sampah', 'payah', 'gagal', 'kecewa',
    'mengecewakan', 'membosankan', 'bosan', 'menjemukan', 'lambat', 'lemah',
    'hancur', 'murahan', 'muak', 'jijik', 'busuk', 'terburuk', 'percuma',
    'rugi', 'benci', 'membenci', 'menyebalkan', 'kesal', 'frustasi',
    'konyol', 'bodoh', 'tolol', 'membingungkan', 'absurd', 'norak', 'klise',
    'picisan', 'dangkal', 'hambar', 'datar', 'prediktabel', 'sia-sia',
}

def translate_indonesian_to_english(text):
    """Translate Indonesian text to English using built-in dictionary."""
    text = text.lower()
    # Try multi-word phrases first
    for id_phrase, en_word in sorted(ID_TO_EN_DICT.items(), key=lambda x: -len(x[0])):
        if ' ' in id_phrase:
            text = text.replace(id_phrase, en_word)
    # Then single words
    words = text.split()
    translated = []
    for word in words:
        # Remove common suffixes (me-, ber-, ter-, ke-, pe-, -kan, -an, -i)
        translated_word = ID_TO_EN_DICT.get(word, None)
        if translated_word is None:
            # Try stripping common affixes
            for prefix in ['me', 'ber', 'ter', 'ke', 'pe', 'di', 'se']:
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    root = word[len(prefix):]
                    translated_word = ID_TO_EN_DICT.get(root)
                    if translated_word:
                        break
            if translated_word is None:
                for suffix in ['kan', 'an', 'nya', 'ku', 'mu']:
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:
                        root = word[:-len(suffix)]
                        translated_word = ID_TO_EN_DICT.get(root)
                        if translated_word:
                            break
        translated.append(translated_word if translated_word else word)
    return ' '.join(translated)

def clean_text_indonesian(text):
    """Clean and preprocess Indonesian text."""
    # 1. HTML Stripping
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Case Folding
    text = text.lower()
    # 3. Remove non-alphabetic
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 4. Tokenization & Stopword Removal (Indonesian)
    words = text.split()
    cleaned_words = [word for word in words if word not in INDONESIAN_STOPWORDS and len(word) > 1]
    return ' '.join(cleaned_words)

def is_matching_language(text, expected_lang):
    """
    Check if the input text matches the expected language filter.
    Returns True if matches, False otherwise.
    """
    # Normalize text by removing non-alphabetic characters
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    words = cleaned.split()
    if not words:
        return True

    # Count Indonesian indicators
    id_indicators = INDONESIAN_STOPWORDS.union(set(ID_TO_EN_DICT.keys()))
    id_count = sum(1 for w in words if w in id_indicators)

    # Count English indicators
    en_indicators = stop_words.union({
        'good', 'great', 'love', 'like', 'awesome', 'excellent', 'beautiful', 
        'wonderful', 'brilliant', 'fantastic', 'best', 'nice', 'enjoy', 'cool', 
        'masterpiece', 'perfect', 'amazing', 'fun', 'happy', 'glad', 'highly', 
        'recommend', 'superb', 'entertaining', 'classic', 'gem', 'outstanding',
        'bad', 'terrible', 'worst', 'hate', 'dislike', 'boring', 'waste', 'poor', 
        'awful', 'stupid', 'dumb', 'horrible', 'worse', 'annoying', 'crap', 
        'rubbish', 'fail', 'disappointed', 'dreadful', 'laughable', 'mess', 
        'garbage', 'suck', 'sucks', 'pathetic', 'pointless', 'useless', 'slow',
        'movie', 'film', 'acting', 'actor', 'actress', 'director', 'scene', 'plot',
        'was', 'were', 'had', 'have', 'has', 'are', 'not', 'but', 'or', 'so',
        'very', 'really', 'just', 'more', 'about', 'would', 'one', 'all', 'out',
        'gorgeous', 'top', 'notch', 'cinema', 'story', 'soundtrack', 'effects',
        'visuals', 'character', 'characters', 'performance', 'screen', 'audience',
        'this', 'that', 'it', 'they', 'he', 'she', 'you', 'me', 'my', 'their'
    })
    en_count = sum(1 for w in words if w in en_indicators)

    print(f"[LangCheck] input='{text[:30]}...' expected={expected_lang} | id_count={id_count}, en_count={en_count}")

    # If expected_lang is 'id', but it has way more English words and almost zero Indonesian words
    if expected_lang == 'id':
        if en_count > id_count and id_count <= len(words) * 0.20:
            return False
    # If expected_lang is 'en', but it has way more Indonesian words and almost zero English words
    elif expected_lang == 'en':
        if id_count > en_count and en_count <= len(words) * 0.20:
            return False

    return True

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

def mock_predict(text, language='en'):
    """
    A sophisticated rule-based classifier to act as a fallback/demo mode.
    Supports both English ('en') and Indonesian ('id') input.
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

    if language == 'id':
        # Use Indonesian sentiment dictionaries directly
        raw_words = text.lower().split()
        pos_count = sum(1 for w in raw_words if w in IDN_POSITIVE_WORDS)
        neg_count = sum(1 for w in raw_words if w in IDN_NEGATIVE_WORDS)
        # Also check via translated text
        translated = translate_indonesian_to_english(text)
        cleaned = clean_text(translated)
        words = cleaned.split()
        stemmed_pos = {stemmer.stem(w) for w in positive_words}
        stemmed_neg = {stemmer.stem(w) for w in negative_words}
        pos_count += sum(1 for w in words if w in stemmed_pos)
        neg_count += sum(1 for w in words if w in stemmed_neg)
    else:
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

    # Detect selected language ('en' for English, 'id' for Indonesian)
    language = data.get('language', 'en')

    # Validate if input matches the selected language filter
    if not is_matching_language(review_text, language):
        expected_lang_name = "Bahasa Indonesia" if language == 'id' else "Bahasa Inggris"
        other_lang_name = "Bahasa Inggris" if language == 'id' else "Bahasa Indonesia"
        return jsonify({
            'status': 'error',
            'error_type': 'language_mismatch',
            'message': f'Ulasan tidak sesuai dengan filter bahasa yang dipilih ({expected_lang_name}). Silakan ubah filter ke {other_lang_name} atau sesuaikan ulasan Anda.'
        }), 400

    # If Indonesian, translate to English before ML pipeline
    text_for_ml = review_text
    if language == 'id':
        text_for_ml = translate_indonesian_to_english(review_text)
        print(f"[ID→EN] Original: '{review_text}' | Translated: '{text_for_ml}'")
        
    if demo_mode:
        sentiment, confidence = mock_predict(review_text, language=language)
    else:
        try:
            # 1. Clean the (translated) review text
            cleaned = clean_text(text_for_ml)
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
            sentiment, confidence = mock_predict(review_text, language=language)
            
    return jsonify({
        'status': 'success',
        'sentiment': sentiment,
        'confidence': confidence,
        'demo_mode': demo_mode,
        'language': language
    })

if __name__ == '__main__':
    # Bind to all interfaces for easy access
    app.run(debug=True, host='0.0.0.0', port=5000)
