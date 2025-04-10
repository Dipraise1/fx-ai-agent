import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import datetime
import logging

# Initialize NLTK downloads
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

class NewsAnalyzer:
    """Sentiment analysis model for financial news data"""
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize the sentiment analysis model
        
        Args:
            model_type (str): Type of model to use ('naive_bayes', 'logistic_regression', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.is_trained = False
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add financial specific stop words
        self.financial_stop_words = {
            'market', 'stock', 'stocks', 'index', 'indices', 'trading',
            'trader', 'traders', 'invest', 'investor', 'investors', 'investment',
            'finance', 'financial', 'economy', 'economic', 'bank', 'banks',
            'share', 'shares', 'company', 'companies', 'exchange', 'exchanges',
            'report', 'reports', 'reported', 'dollar', 'dollars', 'euro',
            'pound', 'yen', 'currency', 'currencies', 'market', 'markets',
            'trade', 'trades', 'trading', 'day', 'week', 'month', 'year'
        }
        self.stop_words.update(self.financial_stop_words)
        
        # Create models directory if it doesn't exist
        os.makedirs('models_data', exist_ok=True)
        
        # Model file path
        self.model_file = f"models_data/sentiment_{model_type}_model.joblib"
        self.vectorizer_file = f"models_data/sentiment_vectorizer.joblib"
        
        # Logger
        self.logger = logging.getLogger('trading_agent.sentiment_analysis')
        
        # Try to load pre-trained model
        self._load_model()
        
        # Financial sentiment lexicon
        self.financial_lexicon = self._load_financial_lexicon()
    
    def _load_financial_lexicon(self):
        """
        Load financial sentiment lexicon
        
        Returns:
            dict: Dictionary mapping words to sentiment scores
        """
        # Financial bull/bear terms
        bull_terms = {
            'bullish': 0.8, 'bull': 0.7, 'upside': 0.6, 'rally': 0.7, 'surge': 0.8, 
            'gain': 0.5, 'gains': 0.5, 'increase': 0.4, 'increasing': 0.4, 
            'grow': 0.4, 'growing': 0.4, 'outperform': 0.7, 'outperforming': 0.7,
            'strong': 0.5, 'strength': 0.5, 'positive': 0.6, 'recovery': 0.6,
            'breakout': 0.7, 'support': 0.4, 'rebound': 0.6, 'momentum': 0.4,
            'opportunity': 0.5, 'opportunities': 0.5, 'buy': 0.6, 'buying': 0.6,
            'upgrade': 0.7, 'upgraded': 0.7, 'raise': 0.5, 'raised': 0.5,
            'beat': 0.6, 'better-than-expected': 0.7, 'exceed': 0.6, 'exceeded': 0.6,
            'above': 0.4, 'high': 0.4, 'higher': 0.4, 'boom': 0.8, 'soar': 0.8,
            'record': 0.5, 'records': 0.5, 'optimistic': 0.7, 'optimism': 0.7,
            'confidence': 0.6, 'confident': 0.6, 'hawkish': 0.5, 'uptick': 0.5,
            'uptrend': 0.7, 'growth': 0.6, 'expand': 0.5, 'expansion': 0.5,
            'breakthrough': 0.7, 'profit': 0.6, 'profitable': 0.6, 'robust': 0.6,
            'succeed': 0.5, 'success': 0.5, 'successful': 0.5
        }
        
        bear_terms = {
            'bearish': -0.8, 'bear': -0.7, 'downside': -0.6, 'sell-off': -0.7, 'selloff': -0.7, 
            'sink': -0.7, 'sank': -0.7, 'loss': -0.5, 'losses': -0.5, 'decrease': -0.4, 
            'decreasing': -0.4, 'decline': -0.5, 'declining': -0.5, 'shrink': -0.4, 
            'shrinking': -0.4, 'underperform': -0.7, 'underperforming': -0.7,
            'weak': -0.5, 'weakness': -0.5, 'negative': -0.6, 'crash': -0.9,
            'crisis': -0.8, 'risk': -0.5, 'risks': -0.5, 'risky': -0.6,
            'recession': -0.8, 'downturn': -0.7, 'slump': -0.6, 'resistance': -0.4,
            'worry': -0.5, 'worried': -0.5, 'fear': -0.6, 'feared': -0.6,
            'sell': -0.6, 'selling': -0.6, 'downgrade': -0.7, 'downgraded': -0.7,
            'cut': -0.5, 'cuts': -0.5, 'miss': -0.6, 'missed': -0.6,
            'worse-than-expected': -0.7, 'below': -0.4, 'low': -0.4, 'lower': -0.4,
            'fall': -0.5, 'falling': -0.5, 'fell': -0.5, 'drop': -0.5,
            'dropping': -0.5, 'dropped': -0.5, 'plunge': -0.8, 'plunging': -0.8,
            'plunged': -0.8, 'pessimistic': -0.7, 'pessimism': -0.7,
            'uncertainty': -0.6, 'uncertain': -0.6, 'cautious': -0.4, 'caution': -0.4,
            'warning': -0.6, 'warn': -0.6, 'warned': -0.6, 'bearish': -0.7,
            'dovish': -0.5, 'downtick': -0.5, 'downtrend': -0.7, 'contraction': -0.6,
            'contract': -0.5, 'concern': -0.5, 'concerns': -0.5, 'concerned': -0.5,
            'fail': -0.6, 'failed': -0.6, 'failure': -0.6, 'default': -0.8
        }
        
        # FX specific terms
        fx_bull_terms = {
            'strengthen': 0.6, 'strengthening': 0.6, 'stronger': 0.5, 'appreciation': 0.6,
            'appreciate': 0.6, 'appreciating': 0.6, 'hawkish': 0.6, 'hike': 0.5, 
            'hiked': 0.5, 'hiking': 0.5, 'intervention': 0.4, 'intervene': 0.4
        }
        
        fx_bear_terms = {
            'weaken': -0.6, 'weakening': -0.6, 'weaker': -0.5, 'depreciation': -0.6,
            'depreciate': -0.6, 'depreciating': -0.6, 'dovish': -0.6, 'cut': -0.5,
            'cutting': -0.5, 'intervention': -0.4, 'intervene': -0.4
        }
        
        # Merge all lexicons
        lexicon = {}
        lexicon.update(bull_terms)
        lexicon.update(bear_terms)
        lexicon.update(fx_bull_terms)
        lexicon.update(fx_bear_terms)
        
        return lexicon
    
    def _preprocess_text(self, text):
        """
        Preprocess text data
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize and remove stop words
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Join tokens back to text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def train(self, df, text_column='text', label_column='sentiment', test_size=0.2):
        """
        Train the sentiment analysis model
        
        Args:
            df (DataFrame): DataFrame with news data
            text_column (str): Column name for news text
            label_column (str): Column name for sentiment labels
            test_size (float): Proportion of data to use for testing
            
        Returns:
            float: Accuracy on test data
        """
        # Preprocess text
        df['processed_text'] = df[text_column].apply(self._preprocess_text)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df[label_column], test_size=test_size, random_state=42
        )
        
        # Create TF-IDF features
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Initialize the model based on model_type
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train the model
        self.model.fit(X_train_vec, y_train)
        
        # Make predictions on the test set
        y_pred = self.model.predict(X_test_vec)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        self.logger.info(f"Model trained: sentiment_{self.model_type}")
        self.logger.info(f"Test accuracy: {accuracy:.4f}")
        self.logger.info(f"Classification report:\n{report}")
        
        # Save the model and vectorizer
        self._save_model()
        
        self.is_trained = True
        
        return accuracy
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a given text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary with sentiment prediction information
        """
        if not self.is_trained and self.model is None:
            # If model not trained, use lexicon-based approach
            return self._lexicon_based_sentiment(text)
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Convert to TF-IDF features
        text_vec = self.vectorizer.transform([processed_text])
        
        # Make prediction
        sentiment_label = self.model.predict(text_vec)[0]
        
        # Get prediction probability
        proba = self.model.predict_proba(text_vec)[0]
        confidence = float(max(proba))
        
        # Use lexicon for additional analysis
        lexicon_sentiment = self._lexicon_based_sentiment(text)
        
        # Combine model and lexicon results
        result = {
            "sentiment": sentiment_label,
            "confidence": confidence,
            "score": lexicon_sentiment["score"],
            "bull_terms": lexicon_sentiment["bull_terms"],
            "bear_terms": lexicon_sentiment["bear_terms"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return result
    
    def _lexicon_based_sentiment(self, text):
        """
        Calculate sentiment based on financial lexicon
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary with sentiment information
        """
        # Convert to lowercase and tokenize
        if not isinstance(text, str):
            return {"sentiment": "neutral", "score": 0.0, "bull_terms": [], "bear_terms": [], "confidence": 0.5}
            
        text = text.lower()
        tokens = word_tokenize(text)
        
        # Find matches in lexicon
        sentiment_score = 0.0
        matched_bull_terms = []
        matched_bear_terms = []
        
        for token in tokens:
            if token in self.financial_lexicon:
                score = self.financial_lexicon[token]
                sentiment_score += score
                
                if score > 0:
                    matched_bull_terms.append(token)
                else:
                    matched_bear_terms.append(token)
        
        # Determine sentiment based on score
        if sentiment_score > 0.5:
            sentiment = "bullish"
            confidence = min(0.5 + abs(sentiment_score) / 10, 0.95)
        elif sentiment_score < -0.5:
            sentiment = "bearish"
            confidence = min(0.5 + abs(sentiment_score) / 10, 0.95)
        else:
            sentiment = "neutral"
            confidence = 0.5
            
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "bull_terms": matched_bull_terms,
            "bear_terms": matched_bear_terms,
            "confidence": confidence
        }
    
    def analyze_news_batch(self, news_items):
        """
        Analyze a batch of news items
        
        Args:
            news_items (list): List of news items (dictionaries with 'title', 'summary', etc.)
            
        Returns:
            dict: Dictionary with sentiment analysis results
        """
        if not news_items:
            return {
                "overall_sentiment": "neutral", 
                "score": 0.0, 
                "confidence": 0.0,
                "items": []
            }
            
        analyzed_items = []
        overall_score = 0.0
        bull_count = 0
        bear_count = 0
        neutral_count = 0
        
        for item in news_items:
            # Combine title and summary for analysis
            title = item.get('headline', '') or item.get('title', '')
            summary = item.get('summary', '') or item.get('description', '')
            
            combined_text = f"{title}. {summary}"
            
            # Analyze sentiment
            sentiment_result = self.predict_sentiment(combined_text)
            
            # Count sentiments
            if sentiment_result['sentiment'] == 'bullish':
                bull_count += 1
            elif sentiment_result['sentiment'] == 'bearish':
                bear_count += 1
            else:
                neutral_count += 1
                
            overall_score += sentiment_result['score']
            
            # Add sentiment to item
            analyzed_item = {
                'title': title,
                'summary': summary,
                'datetime': item.get('datetime', None),
                'url': item.get('url', None),
                'source': item.get('source', None),
                'sentiment': sentiment_result['sentiment'],
                'score': sentiment_result['score'],
                'confidence': sentiment_result['confidence'],
                'bull_terms': sentiment_result['bull_terms'],
                'bear_terms': sentiment_result['bear_terms']
            }
            
            analyzed_items.append(analyzed_item)
            
        # Calculate overall sentiment
        if bull_count > bear_count and bull_count > neutral_count:
            overall_sentiment = "bullish"
        elif bear_count > bull_count and bear_count > neutral_count:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
            
        # Calculate overall confidence
        total_items = len(news_items)
        if overall_sentiment == "bullish":
            confidence = bull_count / total_items
        elif overall_sentiment == "bearish":
            confidence = bear_count / total_items
        else:
            confidence = neutral_count / total_items
            
        # Normalize overall score
        overall_score = overall_score / total_items if total_items > 0 else 0.0
            
        return {
            "overall_sentiment": overall_sentiment,
            "score": overall_score,
            "confidence": confidence,
            "bull_count": bull_count,
            "bear_count": bear_count,
            "neutral_count": neutral_count,
            "total_items": total_items,
            "items": analyzed_items,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _save_model(self):
        """Save model and vectorizer to disk"""
        if self.model is not None:
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.vectorizer, self.vectorizer_file)
            self.logger.info(f"Model saved to {self.model_file}")
            
    def _load_model(self):
        """Load model and vectorizer from disk if they exist"""
        try:
            if os.path.exists(self.model_file):
                self.model = joblib.load(self.model_file)
                self.vectorizer = joblib.load(self.vectorizer_file)
                self.is_trained = True
                self.logger.info(f"Model loaded from {self.model_file}")
                return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
        return False 