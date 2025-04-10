import numpy as np
import pandas as pd
from src.trade_setup import TradeSetup
from src.models.price_prediction import PricePredictionModel
from src.models.sentiment_analysis import NewsAnalyzer

class CryptoStrategy:
    """
    Cryptocurrency Trading Strategy for SOL/USD and ETH/USD
    
    Timeframe: 15m-1h
    Indicators: Moving Averages, RSI, Bollinger Bands, Price Prediction Model
    """
    
    def __init__(self, data_fetcher):
        """Initialize strategy with data fetcher"""
        self.data_fetcher = data_fetcher
        self.name = "Crypto_Strategy"
        self.instruments = ["SOLUSD", "ETHUSD"]
        
        # Initialize price prediction models
        self.price_models = {}
        for instrument in self.instruments:
            self.price_models[instrument] = PricePredictionModel(
                instrument=instrument, 
                timeframe='1h', 
                model_type='random_forest'
            )
            
        # Initialize sentiment analyzer
        self.sentiment_analyzer = NewsAnalyzer(model_type='naive_bayes')
        
    def analyze(self):
        """
        Analyze cryptocurrency pairs for trading opportunities
        
        Returns:
            list of TradeSetup objects with trade details or no-trade reason
        """
        setups = []
        
        # Analyze each instrument
        for instrument in self.instruments:
            setup = self._analyze_instrument(instrument)
            if setup:
                setups.append(setup)
                
        return setups
    
    def _analyze_instrument(self, instrument):
        """
        Analyze a specific cryptocurrency instrument
        
        Args:
            instrument: Cryptocurrency symbol (e.g., 'SOLUSD', 'ETHUSD')
            
        Returns:
            TradeSetup object with trade details or no-trade reason
        """
        # Get data for different timeframes
        df_15m = self.data_fetcher.get_forex_data(instrument, resolution='15', count=100)
        df_1h = self.data_fetcher.get_forex_data(instrument, resolution='60', count=48)
        df_4h = self.data_fetcher.get_forex_data(instrument, resolution='240', count=30)
        
        # Add technical indicators
        if not df_15m.empty and not df_1h.empty and not df_4h.empty:
            df_15m = self.data_fetcher.add_technical_indicators(df_15m)
            df_1h = self.data_fetcher.add_technical_indicators(df_1h)
            df_4h = self.data_fetcher.add_technical_indicators(df_4h)
            
            # Get latest prices
            current_price = df_15m['close'].iloc[-1]
            
            # Get news for sentiment analysis
            news_items = self.data_fetcher.get_news(category="crypto", min_id=0)
            
            # Filter news for this specific cryptocurrency
            crypto_name = "Solana" if instrument == "SOLUSD" else "Ethereum"
            relevant_news = [
                item for item in news_items 
                if crypto_name.lower() in (item.get('headline', '') or item.get('summary', '')).lower()
            ][:5]  # Get up to 5 relevant news items
            
            # Get sentiment from news
            if relevant_news:
                sentiment_results = self.sentiment_analyzer.analyze_news_batch(relevant_news)
                market_sentiment = sentiment_results['overall_sentiment']
                sentiment_score = sentiment_results['score']
                sentiment_confidence = sentiment_results['confidence']
            else:
                market_sentiment = "neutral"
                sentiment_score = 0
                sentiment_confidence = 0
                
            # Train price prediction model if not trained
            if not self.price_models[instrument].is_trained and not df_1h.empty:
                self.price_models[instrument].train(df_1h)
                
            # Get price forecast if model is trained
            if self.price_models[instrument].is_trained:
                forecast = self.price_models[instrument].get_forecast(df_1h, n_future=5)
                forecast_direction = forecast['forecast_direction']
                forecast_confidence = forecast['confidence']
            else:
                forecast_direction = "neutral"
                forecast_confidence = 0
                
            # Check for trade setups
            return self._check_for_setups(
                instrument, 
                df_15m, 
                df_1h, 
                df_4h, 
                market_sentiment,
                sentiment_score,
                forecast_direction,
                forecast_confidence
            )
        else:
            return TradeSetup.no_trade(
                instrument=instrument,
                strategy=self.name,
                reason="Unable to fetch required market data"
            )
            
    def _check_for_setups(self, instrument, df_15m, df_1h, df_4h, 
                          market_sentiment, sentiment_score,
                          forecast_direction, forecast_confidence):
        """
        Check for specific trade setups based on the strategy criteria
        
        Args:
            instrument: Cryptocurrency symbol
            df_15m: 15m data with indicators
            df_1h: 1h data with indicators
            df_4h: 4h data with indicators
            market_sentiment: Market sentiment from news
            sentiment_score: Sentiment score from news
            forecast_direction: Price forecast direction
            forecast_confidence: Price forecast confidence
            
        Returns:
            TradeSetup with trade details or no-trade reason
        """
        # Get latest data points
        latest_15m = df_15m.iloc[-1]
        latest_1h = df_1h.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        
        # Current price
        current_price = latest_15m['close']
        
        # Check for trend alignment in multiple timeframes
        trend_15m = self._determine_trend(df_15m)
        trend_1h = self._determine_trend(df_1h)
        trend_4h = self._determine_trend(df_4h)
        
        # Determine overall trend direction
        if trend_15m == trend_1h == trend_4h and trend_15m != "neutral":
            overall_trend = trend_15m
            trend_strength = "strong"
        elif trend_1h == trend_4h and trend_1h != "neutral":
            overall_trend = trend_1h
            trend_strength = "moderate"
        elif trend_4h != "neutral":
            overall_trend = trend_4h
            trend_strength = "weak"
        else:
            overall_trend = "neutral"
            trend_strength = "none"
            
        # Check for RSI divergence or extreme readings
        rsi_signal = self._check_rsi(df_15m, df_1h)
        
        # Check for Bollinger Band setups
        bb_signal = self._check_bollinger_bands(df_15m, df_1h)
        
        # Combine technical signals with sentiment and forecast
        trade_direction = self._determine_trade_direction(
            overall_trend, trend_strength, rsi_signal, bb_signal,
            market_sentiment, forecast_direction
        )
        
        # If no clear direction, no trade
        if trade_direction == "neutral":
            return TradeSetup.no_trade(
                instrument=instrument,
                strategy=self.name,
                reason=f"No clear trading signal. Trend: {overall_trend}, RSI: {rsi_signal}, BB: {bb_signal}, Sentiment: {market_sentiment}"
            )
            
        # Calculate entry zone, stop loss and targets based on ATR
        atr = latest_1h.get('atr', current_price * 0.01)  # Default to 1% if ATR not available
        
        if trade_direction == "Long":
            entry_zone = (current_price, current_price * (1 + 0.005))  # 0.5% range
            stop_loss = current_price - atr * 1.5
            targets = [current_price + atr * 2, current_price + atr * 3.5]
        else:  # Short
            entry_zone = (current_price * (1 - 0.005), current_price)  # 0.5% range
            stop_loss = current_price + atr * 1.5
            targets = [current_price - atr * 2, current_price - atr * 3.5]
            
        # Calculate risk/reward
        risk_reward = TradeSetup.calculate_risk_reward(
            entry=current_price,
            stop=stop_loss,
            target=targets[0]
        )
        
        # Determine confidence level (0-100%)
        confidence = self._calculate_confidence(
            trade_direction, overall_trend, trend_strength, 
            rsi_signal, bb_signal, market_sentiment, sentiment_score,
            forecast_direction, forecast_confidence, risk_reward
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            instrument, trade_direction, overall_trend, trend_strength,
            rsi_signal, bb_signal, market_sentiment, forecast_direction
        )
        
        # Return trade setup
        return TradeSetup(
            instrument=instrument,
            direction=trade_direction,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            targets=targets,
            risk_reward=risk_reward,
            confidence=confidence,
            rationale=rationale,
            strategy=self.name
        )
    
    def _determine_trend(self, df):
        """Determine trend direction based on moving averages"""
        if 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            return "neutral"
            
        latest = df.iloc[-1]
        sma20 = latest['sma_20']
        sma50 = latest['sma_50']
        close = latest['close']
        
        # Check trend based on moving average relationship
        if close > sma20 > sma50:
            return "Long"
        elif close < sma20 < sma50:
            return "Short"
        else:
            return "neutral"
    
    def _check_rsi(self, df_15m, df_1h):
        """Check RSI for divergence or extreme readings"""
        if 'rsi' not in df_15m.columns or 'rsi' not in df_1h.columns:
            return "neutral"
            
        latest_15m = df_15m.iloc[-1]
        latest_1h = df_1h.iloc[-1]
        
        rsi_15m = latest_15m['rsi']
        rsi_1h = latest_1h['rsi']
        
        # Check for oversold condition (RSI < 30)
        if rsi_15m < 30 and rsi_1h < 40:
            return "Long"
            
        # Check for overbought condition (RSI > 70)
        elif rsi_15m > 70 and rsi_1h > 60:
            return "Short"
            
        return "neutral"
    
    def _check_bollinger_bands(self, df_15m, df_1h):
        """Check Bollinger Bands for price touching or crossing bands"""
        if 'BBL_20_2.0' not in df_15m.columns or 'BBU_20_2.0' not in df_15m.columns:
            return "neutral"
            
        latest_15m = df_15m.iloc[-1]
        recent_15m = df_15m.iloc[-3:]
        
        lower_band = latest_15m['BBL_20_2.0']
        upper_band = latest_15m['BBU_20_2.0']
        close = latest_15m['close']
        
        # Check for price touching lower band (potential long)
        lower_touch = any(recent_15m['low'] <= recent_15m['BBL_20_2.0'])
        
        # Check for price touching upper band (potential short)
        upper_touch = any(recent_15m['high'] >= recent_15m['BBU_20_2.0'])
        
        if lower_touch and close > lower_band:
            return "Long"
        elif upper_touch and close < upper_band:
            return "Short"
            
        return "neutral"
    
    def _determine_trade_direction(self, overall_trend, trend_strength, rsi_signal, 
                                   bb_signal, market_sentiment, forecast_direction):
        """Determine overall trade direction based on all signals"""
        # Convert signals to numerical values
        signal_values = {
            "Long": 1,
            "neutral": 0,
            "Short": -1,
            "bullish": 1,
            "bearish": -1
        }
        
        # Assign weights to different signals
        weights = {
            "trend": 0.3,
            "rsi": 0.2,
            "bb": 0.2,
            "sentiment": 0.1,
            "forecast": 0.2
        }
        
        # Adjust trend weight based on strength
        if trend_strength == "strong":
            weights["trend"] = 0.4
            weights["rsi"] = 0.15
            weights["bb"] = 0.15
        elif trend_strength == "weak":
            weights["trend"] = 0.2
            weights["forecast"] = 0.3
            
        # Calculate weighted signal
        weighted_signal = (
            weights["trend"] * signal_values.get(overall_trend, 0) +
            weights["rsi"] * signal_values.get(rsi_signal, 0) +
            weights["bb"] * signal_values.get(bb_signal, 0) +
            weights["sentiment"] * signal_values.get(market_sentiment, 0) +
            weights["forecast"] * signal_values.get(forecast_direction, 0)
        )
        
        # Determine direction based on weighted signal
        if weighted_signal > 0.2:
            return "Long"
        elif weighted_signal < -0.2:
            return "Short"
        else:
            return "neutral"
    
    def _calculate_confidence(self, trade_direction, overall_trend, trend_strength, 
                              rsi_signal, bb_signal, market_sentiment, sentiment_score,
                              forecast_direction, forecast_confidence, risk_reward):
        """Calculate confidence level for the trade setup"""
        base_confidence = 50  # Start with base confidence of 50%
        
        # Add confidence based on trend alignment
        if overall_trend == trade_direction:
            if trend_strength == "strong":
                base_confidence += 15
            elif trend_strength == "moderate":
                base_confidence += 10
            else:
                base_confidence += 5
                
        # Add confidence based on RSI signal
        if rsi_signal == trade_direction:
            base_confidence += 10
            
        # Add confidence based on BB signal
        if bb_signal == trade_direction:
            base_confidence += 10
            
        # Add confidence based on sentiment alignment
        sentiment_match = (
            (trade_direction == "Long" and market_sentiment == "bullish") or
            (trade_direction == "Short" and market_sentiment == "bearish")
        )
        if sentiment_match:
            base_confidence += 5 + min(5, abs(sentiment_score) * 10)
            
        # Add confidence based on forecast alignment
        forecast_match = (
            (trade_direction == "Long" and forecast_direction == "bullish") or
            (trade_direction == "Short" and forecast_direction == "bearish")
        )
        if forecast_match:
            base_confidence += forecast_confidence / 10
            
        # Add confidence based on risk/reward ratio
        if risk_reward:
            if risk_reward >= 3:
                base_confidence += 15
            elif risk_reward >= 2:
                base_confidence += 10
            elif risk_reward >= 1.5:
                base_confidence += 5
                
        # Ensure confidence is within 0-100 range
        return min(100, max(0, int(base_confidence)))
    
    def _generate_rationale(self, instrument, trade_direction, overall_trend, trend_strength,
                            rsi_signal, bb_signal, market_sentiment, forecast_direction):
        """Generate rationale for the trade setup"""
        crypto_name = "Solana" if instrument == "SOLUSD" else "Ethereum"
        
        rationale = f"{crypto_name} shows a {trend_strength} {overall_trend.lower()} trend across multiple timeframes. "
        
        if rsi_signal != "neutral":
            if rsi_signal == "Long":
                rationale += f"RSI indicates oversold conditions, suggesting potential upward reversal. "
            else:
                rationale += f"RSI indicates overbought conditions, suggesting potential downward reversal. "
                
        if bb_signal != "neutral":
            if bb_signal == "Long":
                rationale += f"Price recently touched the lower Bollinger Band and bounced, indicating potential upward movement. "
            else:
                rationale += f"Price recently touched the upper Bollinger Band and rejected, indicating potential downward movement. "
                
        if market_sentiment != "neutral":
            rationale += f"Market sentiment is {market_sentiment} based on recent news. "
            
        if forecast_direction != "neutral":
            rationale += f"Price prediction model suggests {forecast_direction} movement in the near future. "
            
        return rationale 