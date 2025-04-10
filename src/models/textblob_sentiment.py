"""
TextBlob-based Sentiment Analysis

A lightweight sentiment analysis module that uses TextBlob instead of NLTK.
TextBlob provides easy-to-use sentiment analysis with minimal dependencies.
"""

import logging
import json
from datetime import datetime
from pathlib import Path

try:
    from textblob import TextBlob
except ImportError:
    logging.warning("TextBlob not installed. Please install with: pip install textblob")
    # Create a fallback TextBlob implementation with neutral sentiment
    class FakeTextBlob:
        def __init__(self, text):
            self.text = text
            
        @property
        def sentiment(self):
            class Sentiment:
                def __init__(self):
                    self.polarity = 0.0  # Neutral
                    self.subjectivity = 0.5  # Neutral
            return Sentiment()
            
        @property
        def words(self):
            return self.text.split()
            
    TextBlob = FakeTextBlob

class SentimentAnalyzer:
    """
    TextBlob-based sentiment analyzer for financial and market texts
    """
    
    def __init__(self, cache_dir=None):
        """
        Initialize sentiment analyzer
        
        Args:
            cache_dir: Directory to cache sentiment results
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Sentiment cache directory: {self.cache_dir}")
    
    def analyze_text(self, text, use_cache=True):
        """
        Analyze the sentiment of a text
        
        Args:
            text: Text to analyze
            use_cache: Whether to use cached results
            
        Returns:
            dict: Sentiment analysis results with polarity, subjectivity, and label
        """
        if not text:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'label': 'neutral',
                'text': text,
                'timestamp': datetime.now().isoformat()
            }
        
        # Check cache if enabled
        cache_hit = False
        cache_file = None
        
        if use_cache and self.cache_dir:
            # Create a cache key from the text (simplified hash)
            cache_key = str(hash(text))
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        result = json.load(f)
                    cache_hit = True
                    self.logger.debug(f"Sentiment cache hit for text: {text[:30]}...")
                    return result
                except Exception as e:
                    self.logger.warning(f"Error reading sentiment cache: {e}")
        
        # Perform sentiment analysis
        try:
            # Create TextBlob
            blob = TextBlob(text)
            
            # Get polarity (-1 to 1) and subjectivity (0 to 1)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Create result dictionary
            result = {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': label,
                'text': text,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result if enabled
            if use_cache and self.cache_dir and cache_file:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(result, f)
                except Exception as e:
                    self.logger.warning(f"Error writing sentiment cache: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {
                'polarity': 0,
                'subjectivity': 0,
                'label': 'neutral',
                'text': text,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def analyze_news(self, news_items, use_cache=True):
        """
        Analyze sentiment of multiple news items
        
        Args:
            news_items: List of news items (dict with 'headline' and 'summary' keys)
            use_cache: Whether to use cached results
            
        Returns:
            dict: Overall sentiment and per-item sentiment
        """
        if not news_items:
            return {
                'overall_sentiment': 'neutral',
                'overall_polarity': 0,
                'item_sentiments': [],
                'count': 0
            }
        
        # Analyze each news item
        item_sentiments = []
        total_polarity = 0
        
        for item in news_items:
            # Combine headline and summary
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            text = f"{headline}. {summary}"
            
            # Analyze sentiment
            sentiment = self.analyze_text(text, use_cache=use_cache)
            
            # Add to results
            item_sentiments.append({
                'headline': headline,
                'sentiment': sentiment['label'],
                'polarity': sentiment['polarity']
            })
            
            # Add to total polarity
            total_polarity += sentiment['polarity']
        
        # Calculate overall sentiment
        avg_polarity = total_polarity / len(news_items) if news_items else 0
        
        if avg_polarity > 0.1:
            overall_sentiment = 'positive'
        elif avg_polarity < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Create result
        result = {
            'overall_sentiment': overall_sentiment,
            'overall_polarity': avg_polarity,
            'item_sentiments': item_sentiments,
            'count': len(news_items)
        }
        
        return result


# Usage example
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sentiment analyzer
    analyzer = SentimentAnalyzer(cache_dir="data/cache/sentiment")
    
    # Test with sample text
    sample_texts = [
        "The market is showing strong bullish momentum with increasing volume.",
        "Economic indicators remain mixed with some positive signs.",
        "Investors are concerned about the potential recession ahead.",
        "The USD/JPY pair is facing resistance at key levels after the rate decision."
    ]
    
    print("\nTesting sentiment analysis with sample texts:")
    for text in sample_texts:
        sentiment = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment['label']}")
        print(f"Polarity: {sentiment['polarity']:.2f}")
        print(f"Subjectivity: {sentiment['subjectivity']:.2f}")
        
    # Test with sample news
    sample_news = [
        {
            "headline": "Fed raises interest rates by 25 basis points",
            "summary": "The Federal Reserve raised interest rates by 25 basis points, as expected, citing concerns about inflation."
        },
        {
            "headline": "Strong jobs report exceeds expectations",
            "summary": "The latest jobs report showed the economy added more jobs than expected, indicating robust economic growth."
        },
        {
            "headline": "Tech stocks decline amid valuation concerns",
            "summary": "Technology stocks declined today as investors raised concerns about high valuations in the sector."
        }
    ]
    
    print("\nTesting news sentiment analysis:")
    news_sentiment = analyzer.analyze_news(sample_news)
    print(f"Overall sentiment: {news_sentiment['overall_sentiment']}")
    print(f"Overall polarity: {news_sentiment['overall_polarity']:.2f}")
    print(f"Analyzed {news_sentiment['count']} news items")
    
    for i, item in enumerate(news_sentiment['item_sentiments']):
        print(f"\nNews {i+1}: {item['headline']}")
        print(f"Sentiment: {item['sentiment']}")
        print(f"Polarity: {item['polarity']:.2f}") 