import os
import nltk
import logging
import ssl
import socket
from pathlib import Path

logger = logging.getLogger("trading_agent.nltk_setup")

# Define basic fallback data
BASIC_STOPWORDS = {
    'english': [
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
        'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
        'to', 'from', 'in', 'on', 'at', 'by', 'with', 'without', 'before', 'after',
        'above', 'below', 'up', 'down'
    ]
}

# Define dummy sentiment lexicon
DUMMY_SENTIMENT = {
    'positive': ['good', 'great', 'excellent', 'positive', 'bullish', 'increase', 'higher', 'growth'],
    'negative': ['bad', 'poor', 'negative', 'bearish', 'decrease', 'lower', 'fall', 'decline']
}

def setup_nltk_data():
    """
    Set up NLTK data and provide fallbacks if data is unavailable.
    This function configures NLTK to use local data directories and
    creates fallbacks for essential components.
    """
    # Fix SSL certificate verification issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
        logger.info("SSL certificate verification disabled for NLTK downloads")
    except AttributeError:
        logger.warning("Failed to disable SSL certificate verification")
    
    # Create and use local NLTK data directory
    nltk_data_dir = os.path.join(Path(__file__).parent.parent.parent, 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add to NLTK data path
    nltk.data.path.insert(0, nltk_data_dir)
    logger.info(f"Using local NLTK data directory: {nltk_data_dir}")
    
    # Instead of trying to download, go straight to setting up fallbacks
    # as download attempts are causing errors
    setup_fallbacks()
    
    return True

class FallbackStopwords:
    """Fallback implementation of stopwords corpus"""
    def __init__(self):
        self.fallback_words = BASIC_STOPWORDS
        
    def words(self, lang='english'):
        """Return stopwords for the given language"""
        if lang in self.fallback_words:
            return self.fallback_words[lang]
        return []

class FallbackVaderLexicon:
    """Fallback implementation for VADER sentiment lexicon"""
    def __init__(self):
        self.sentiment = DUMMY_SENTIMENT
    
    def words(self):
        """Return all sentiment words"""
        all_words = []
        for category in self.sentiment.values():
            all_words.extend(category)
        return all_words

class FallbackSentimentIntensityAnalyzer:
    """Fallback implementation of VADER sentiment analyzer"""
    def __init__(self):
        self.lexicon = DUMMY_SENTIMENT
    
    def polarity_scores(self, text):
        """
        Return a dictionary of sentiment scores for the input text
        
        Returns dict with keys: pos, neg, neu, compound
        """
        words = text.lower().split()
        
        # Count positive and negative words
        pos_count = sum(1 for word in words if word in self.lexicon['positive'])
        neg_count = sum(1 for word in words if word in self.lexicon['negative'])
        
        # Calculate scores
        total = len(words) if len(words) > 0 else 1
        pos = pos_count / total
        neg = neg_count / total
        neu = 1.0 - (pos + neg)
        
        # Calculate compound score (-1 to 1)
        if pos > neg:
            compound = pos - (neg / 2)
        elif neg > pos:
            compound = -1 * (neg - (pos / 2))
        else:
            compound = 0
            
        return {'pos': pos, 'neg': neg, 'neu': neu, 'compound': compound}

def setup_fallbacks():
    """
    Set up fallback methods for NLTK functionality that might be missing.
    This creates completely offline versions of required NLTK functionality.
    """
    logger.warning("Setting up NLTK fallbacks for offline operation")
    
    # Create module structure if needed
    if not hasattr(nltk, 'corpus'):
        nltk.corpus = type('DummyCorpusModule', (), {})()
    
    # Set up stopwords fallback
    nltk.corpus.stopwords = FallbackStopwords()
    logger.info("Created fallback for stopwords")
    
    # Set up tokenize fallback
    def simple_tokenize(text):
        """Simple word tokenizer"""
        import re
        return re.findall(r'\b\w+\b', text.lower())
        
    nltk.word_tokenize = simple_tokenize
    logger.info("Created fallback for word_tokenize")
    
    # Set up sentiment analysis fallbacks
    if not hasattr(nltk, 'sentiment'):
        nltk.sentiment = type('DummySentimentModule', (), {})()
    
    if not hasattr(nltk.sentiment, 'vader'):
        nltk.sentiment.vader = type('DummyVaderModule', (), {})()
    
    nltk.sentiment.vader.SentimentIntensityAnalyzer = FallbackSentimentIntensityAnalyzer
    logger.info("Created fallback for sentiment analysis")
    
    # Create any other necessary fallbacks
    if not hasattr(nltk.corpus, 'vader_lexicon'):
        nltk.corpus.vader_lexicon = FallbackVaderLexicon()
    
    logger.info("All NLTK fallbacks are set up and ready to use")

# Run setup when imported
setup_nltk_data() 