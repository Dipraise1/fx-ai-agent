#!/usr/bin/env python

import pandas as pd
from textblob import TextBlob
import logging

class SentimentAnalyzer:
    """
    A class for analyzing sentiment in text data using TextBlob.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            logger: Logger object for logging. If None, a new logger will be created.
        """
        if logger:
            self.logger = logger
        else:
            # Set up logger
            self.logger = logging.getLogger('SentimentAnalyzer')
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("SentimentAnalyzer initialized")
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of a single text.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            dict: Dictionary containing polarity and subjectivity scores.
        """
        try:
            if not text or not isinstance(text, str):
                self.logger.warning(f"Invalid text input: {text}")
                return {"polarity": 0, "subjectivity": 0}
            
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            return {
                "polarity": sentiment.polarity,  # Range: -1 (negative) to 1 (positive)
                "subjectivity": sentiment.subjectivity  # Range: 0 (objective) to 1 (subjective)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return {"polarity": 0, "subjectivity": 0}
    
    def analyze_dataframe(self, df, text_column):
        """
        Analyze sentiment for texts in a DataFrame column.
        
        Args:
            df (pandas.DataFrame): DataFrame containing text data.
            text_column (str): Name of the column containing text to analyze.
            
        Returns:
            pandas.DataFrame: Original DataFrame with added sentiment columns.
        """
        if text_column not in df.columns:
            self.logger.error(f"Column {text_column} not found in DataFrame")
            return df
        
        self.logger.info(f"Analyzing sentiment for {len(df)} texts")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply sentiment analysis to each text
        sentiments = result_df[text_column].apply(self.analyze_text)
        
        # Extract polarity and subjectivity into separate columns
        result_df['sentiment_polarity'] = sentiments.apply(lambda x: x['polarity'])
        result_df['sentiment_subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])
        
        self.logger.info("Sentiment analysis completed")
        return result_df
    
    def get_sentiment_category(self, polarity):
        """
        Convert polarity score to a sentiment category.
        
        Args:
            polarity (float): Sentiment polarity score (-1 to 1).
            
        Returns:
            str: Sentiment category ('negative', 'neutral', or 'positive').
        """
        if polarity < -0.1:
            return 'negative'
        elif polarity > 0.1:
            return 'positive'
        else:
            return 'neutral'
    
    def calculate_weighted_sentiment(self, texts, weights=None):
        """
        Calculate weighted sentiment across multiple texts.
        
        Args:
            texts (list): List of text strings to analyze.
            weights (list, optional): List of weights for each text. If None, equal weights are used.
            
        Returns:
            dict: Dictionary containing weighted polarity and subjectivity.
        """
        if not texts:
            return {"weighted_polarity": 0, "weighted_subjectivity": 0}
        
        # Use equal weights if none provided
        if weights is None:
            weights = [1/len(texts)] * len(texts)
        
        # Ensure weights sum to 1
        if sum(weights) != 1:
            weights = [w/sum(weights) for w in weights]
        
        if len(weights) != len(texts):
            self.logger.warning("Number of weights doesn't match number of texts, using equal weights")
            weights = [1/len(texts)] * len(texts)
        
        # Analyze each text
        sentiments = [self.analyze_text(text) for text in texts]
        
        # Calculate weighted scores
        weighted_polarity = sum(s["polarity"] * w for s, w in zip(sentiments, weights))
        weighted_subjectivity = sum(s["subjectivity"] * w for s, w in zip(sentiments, weights))
        
        return {
            "weighted_polarity": weighted_polarity,
            "weighted_subjectivity": weighted_subjectivity,
            "sentiment_category": self.get_sentiment_category(weighted_polarity)
        }


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Single text analysis
    text = "The market is showing strong bullish signals with increasing volume."
    result = analyzer.analyze_text(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result}")
    print(f"Category: {analyzer.get_sentiment_category(result['polarity'])}")
    
    # Multiple texts with weights
    texts = [
        "The economy is struggling with inflation concerns.",
        "Tech stocks are rallying after positive earnings.",
        "Investors remain cautious amid geopolitical tensions."
    ]
    weights = [0.2, 0.5, 0.3]
    weighted_result = analyzer.calculate_weighted_sentiment(texts, weights)
    print("\nWeighted sentiment analysis:")
    print(f"Weighted polarity: {weighted_result['weighted_polarity']:.2f}")
    print(f"Weighted subjectivity: {weighted_result['weighted_subjectivity']:.2f}")
    print(f"Sentiment category: {weighted_result['sentiment_category']}") 