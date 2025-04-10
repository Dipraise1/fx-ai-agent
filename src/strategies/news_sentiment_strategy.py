"""
News Sentiment Trading Strategy

This strategy analyzes financial news and generates trade setups based on sentiment analysis.
It uses TextBlob for sentiment analysis rather than NLTK, making it more lightweight.
"""

import os
import logging
import datetime
from pathlib import Path

# Import the TextBlob-based sentiment analyzer
from src.models.textblob_sentiment import SentimentAnalyzer
from src.trade_setup import TradeSetup

class NewsSentimentStrategy:
    """
    Trading strategy based on news sentiment analysis using TextBlob
    """
    
    def __init__(self, data_fetcher):
        """
        Initialize the news sentiment strategy
        
        Args:
            data_fetcher: DataFetcher instance to retrieve market data and news
        """
        self.name = "News_Sentiment"
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentiment analyzer
        cache_dir = Path('data/cache/sentiment')
        os.makedirs(cache_dir, exist_ok=True)
        self.sentiment_analyzer = SentimentAnalyzer(cache_dir=cache_dir)
        
        # Instruments to analyze
        self.instruments = [
            "EURUSD",    # EUR/USD
            "GBPUSD",    # GBP/USD
            "USDJPY",    # USD/JPY
            "ETHUSD",    # ETH/USD
        ]
    
    def analyze(self):
        """
        Analyze market news and generate trade setups
        
        Returns:
            list: List of TradeSetup objects
        """
        self.logger.info("Running News Sentiment Strategy analysis")
        
        # Get recent news for the instruments
        setups = []
        
        for instrument in self.instruments:
            # Get news specific to this instrument
            try:
                news = self._get_instrument_news(instrument)
                if not news:
                    self.logger.info(f"No news found for {instrument}")
                    continue
                
                # Analyze news sentiment
                sentiment_result = self.sentiment_analyzer.analyze_news(news)
                
                # Generate trade setup based on sentiment
                setup = self._generate_setup(instrument, sentiment_result, news)
                if setup:
                    setups.append(setup)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {instrument}: {e}")
        
        # Return all generated setups
        return setups
        
    def _get_instrument_news(self, instrument):
        """
        Get recent news for a specific instrument
        
        Args:
            instrument: Instrument to get news for
            
        Returns:
            list: News items for the instrument
        """
        try:
            # Get news from data fetcher
            news = self.data_fetcher.get_news(category="forex", count=20)
            
            if not news:
                return []
            
            # Filter news for the instrument
            # Extract currency symbols from the instrument
            if instrument == "EURUSD":
                keywords = ["EUR", "USD", "Euro", "Dollar", "EUR/USD", "EURUSD"]
            elif instrument == "GBPUSD":
                keywords = ["GBP", "USD", "Pound", "Sterling", "Dollar", "GBP/USD", "GBPUSD"]
            elif instrument == "USDJPY":
                keywords = ["USD", "JPY", "Dollar", "Yen", "USD/JPY", "USDJPY"]
            elif instrument == "ETHUSD":
                keywords = ["ETH", "USD", "Ethereum", "Dollar", "ETH/USD", "ETHUSD"]
            else:
                keywords = [instrument]
            
            # Filter by keywords in headline or summary
            filtered_news = []
            for item in news:
                headline = item.get('headline', '').upper()
                summary = item.get('summary', '').upper()
                
                # Check if any keyword matches
                if any(kw.upper() in headline or kw.upper() in summary for kw in keywords):
                    filtered_news.append(item)
            
            return filtered_news
            
        except Exception as e:
            self.logger.error(f"Error getting news for {instrument}: {e}")
            return []
    
    def _generate_setup(self, instrument, sentiment_result, news):
        """
        Generate a trade setup based on sentiment analysis
        
        Args:
            instrument: Instrument to generate setup for
            sentiment_result: Results from sentiment analysis
            news: Raw news items
            
        Returns:
            TradeSetup: Generated trade setup or None
        """
        # Get overall sentiment
        overall_sentiment = sentiment_result.get('overall_sentiment', 'neutral')
        overall_polarity = sentiment_result.get('overall_polarity', 0)
        
        # If neutral or no clear signal, don't generate a setup
        if overall_sentiment == 'neutral' or abs(overall_polarity) < 0.2:
            return TradeSetup(
                instrument=instrument,
                direction="No Trade",
                entry_zone=None,
                stop_loss=None,
                targets=[],
                risk_reward=None,
                confidence=0,
                rationale="News sentiment is neutral or unclear",
                strategy=self.name
            )
        
        # Get current market data for the instrument
        try:
            # Format instrument for the data fetcher
            symbol = self._format_instrument_for_data_fetcher(instrument)
            
            # Get recent price data
            price_data = self.data_fetcher.get_forex_data(symbol, resolution="D", count=5)
            
            if price_data is None or price_data.empty:
                return TradeSetup(
                    instrument=instrument,
                    direction="No Trade",
                    entry_zone=None,
                    stop_loss=None,
                    targets=[],
                    risk_reward=None,
                    confidence=0,
                    rationale=f"Unable to fetch price data for {instrument}",
                    strategy=self.name
                )
            
            # Get latest price
            latest_price = price_data['close'].iloc[-1]
            
            # Determine direction based on sentiment
            direction = "Long" if overall_sentiment == "positive" else "Short"
            
            # Calculate entry zone, stop loss, and targets
            entry_zone, stop_loss, targets = self._calculate_levels(
                direction, 
                latest_price,
                price_data
            )
            
            # Calculate confidence based on sentiment strength
            confidence = abs(overall_polarity) * 100
            confidence = min(max(confidence, 0), 100)  # Clamp to 0-100
            
            # Calculate risk/reward
            risk = abs(entry_zone[0] - stop_loss)
            if not risk:
                risk_reward = None
            else:
                reward = abs(targets[0] - entry_zone[0]) if targets else 0
                risk_reward = reward / risk if risk else None
            
            # Generate rationale
            news_headlines = [item.get('headline', '') for item in news[:3]]
            news_summary = "; ".join(news_headlines)
            
            rationale = f"News sentiment for {instrument} is {overall_sentiment} "
            rationale += f"(polarity: {overall_polarity:.2f}). "
            rationale += f"Recent news: {news_summary}"
            
            # Create and return trade setup
            return TradeSetup(
                instrument=instrument,
                direction=direction,
                entry_zone=entry_zone,
                stop_loss=stop_loss,
                targets=targets,
                risk_reward=risk_reward,
                confidence=confidence,
                rationale=rationale,
                strategy=self.name
            )
            
        except Exception as e:
            self.logger.error(f"Error generating setup for {instrument}: {e}")
            return TradeSetup(
                instrument=instrument,
                direction="No Trade",
                entry_zone=None,
                stop_loss=None,
                targets=[],
                risk_reward=None,
                confidence=0,
                rationale=f"Error generating setup: {e}",
                strategy=self.name
            )
    
    def _format_instrument_for_data_fetcher(self, instrument):
        """
        Format instrument string for data fetcher
        """
        instrument_map = {
            "EURUSD": "OANDA:EUR_USD",
            "GBPUSD": "OANDA:GBP_USD",
            "USDJPY": "OANDA:USD_JPY",
            "ETHUSD": "BINANCE:ETHUSDT"
        }
        return instrument_map.get(instrument, instrument)
    
    def _calculate_levels(self, direction, current_price, price_data):
        """
        Calculate entry zone, stop loss, and targets based on price action
        
        Args:
            direction: Trade direction ('Long' or 'Short')
            current_price: Current market price
            price_data: Historical price data
            
        Returns:
            tuple: (entry_zone, stop_loss, targets)
        """
        # Calculate price volatility (average true range)
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        # Simple ATR calculation
        tr = []
        for i in range(1, len(close)):
            # True range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            prev_close = close.iloc[i-1]
            tr.append(max(
                high.iloc[i] - low.iloc[i],
                abs(high.iloc[i] - prev_close),
                abs(low.iloc[i] - prev_close)
            ))
        
        atr = sum(tr) / len(tr) if tr else (high.iloc[-1] - low.iloc[-1])
        
        # Calculate levels based on ATR
        if direction == "Long":
            # For long trades, entry zone is slightly below current price
            entry_zone = (current_price - atr * 0.3, current_price)
            
            # Stop loss is below the entry zone
            stop_loss = entry_zone[0] - atr * 1.0
            
            # Targets are above the entry zone
            target1 = entry_zone[1] + atr * 1.0
            target2 = entry_zone[1] + atr * 2.0
            targets = [target1, target2]
            
        else:  # "Short"
            # For short trades, entry zone is slightly above current price
            entry_zone = (current_price, current_price + atr * 0.3)
            
            # Stop loss is above the entry zone
            stop_loss = entry_zone[1] + atr * 1.0
            
            # Targets are below the entry zone
            target1 = entry_zone[0] - atr * 1.0
            target2 = entry_zone[0] - atr * 2.0
            targets = [target1, target2]
        
        return entry_zone, stop_loss, targets 