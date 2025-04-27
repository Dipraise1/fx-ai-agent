import numpy as np
import pandas as pd
from src.trade_setup import TradeSetup
from src.models.price_prediction import PricePredictionModel
from src.models.sentiment_analysis import NewsAnalyzer

class CryptoStrategy:
    """
    Cryptocurrency Trading Strategy for major crypto pairs
    
    Timeframe: 15m-1h
    Indicators: Moving Averages, RSI, Bollinger Bands, Price Prediction Model
    """
    
    def __init__(self, data_fetcher):
        """Initialize strategy with data fetcher"""
        self.data_fetcher = data_fetcher
        self.name = "Crypto_Strategy"
        # Add more cryptocurrency pairs
        self.instruments = ["SOLUSD", "ETHUSD", "BTCUSD", "ADAUSD", "DOTUSD", "LINKUSD"]
        
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
        # Get data for different timeframes with increased count for better analysis
        df_15m = self.data_fetcher.get_forex_data(instrument, resolution='15', count=150)
        df_1h = self.data_fetcher.get_forex_data(instrument, resolution='60', count=72)
        df_4h = self.data_fetcher.get_forex_data(instrument, resolution='240', count=50)
        
        # Add technical indicators
        if not df_15m.empty and not df_1h.empty and not df_4h.empty:
            try:
                df_15m = self.data_fetcher.add_technical_indicators(df_15m)
                df_1h = self.data_fetcher.add_technical_indicators(df_1h)
                df_4h = self.data_fetcher.add_technical_indicators(df_4h)
                
                # Get latest prices
                current_price = df_15m['close'].iloc[-1]
                
                # Map crypto symbol to full name for news filtering
                crypto_name_map = {
                    "SOLUSD": "Solana",
                    "ETHUSD": "Ethereum",
                    "BTCUSD": "Bitcoin",
                    "ADAUSD": "Cardano",
                    "DOTUSD": "Polkadot",
                    "LINKUSD": "Chainlink"
                }
                crypto_name = crypto_name_map.get(instrument, instrument[:3])
                
                # Get news for sentiment analysis
                news_items = self.data_fetcher.get_news(category="crypto", min_id=0)
                
                # Filter news for this specific cryptocurrency
                relevant_news = [
                    item for item in news_items 
                    if crypto_name.lower() in (item.get('headline', '') or item.get('summary', '')).lower()
                ][:5]  # Get up to 5 relevant news items
                
                # Get sentiment from news
                if relevant_news:
                    try:
                        sentiment_results = self.sentiment_analyzer.analyze_news_batch(relevant_news)
                        market_sentiment = sentiment_results['overall_sentiment']
                        sentiment_score = sentiment_results['score']
                        sentiment_confidence = sentiment_results['confidence']
                    except Exception as e:
                        print(f"Error analyzing sentiment: {e}")
                        market_sentiment = "neutral"
                        sentiment_score = 0
                        sentiment_confidence = 0
                else:
                    market_sentiment = "neutral"
                    sentiment_score = 0
                    sentiment_confidence = 0
                    
                # Try to train price prediction model if not trained
                forecast_direction = "neutral"
                forecast_confidence = 0
                
                try:
                    if instrument in self.price_models and self.price_models[instrument] and not self.price_models[instrument].is_trained and not df_1h.empty:
                        try:
                            self.price_models[instrument].train(df_1h)
                        except Exception as e:
                            print(f"Error training price model for {instrument}: {e}")
                        
                    # Get price forecast if model is trained
                    if instrument in self.price_models and self.price_models[instrument] and self.price_models[instrument].is_trained:
                        forecast = self.price_models[instrument].get_forecast(df_1h, n_future=5)
                        forecast_direction = forecast['forecast_direction']
                        forecast_confidence = forecast['confidence']
                except Exception as e:
                    print(f"Error with price prediction for {instrument}: {e}")
                    
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
            except Exception as e:
                return TradeSetup.no_trade(
                    instrument=instrument,
                    strategy=self.name,
                    reason=f"Error analyzing market data: {str(e)}"
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
        """Determine trend direction based on moving averages and additional criteria"""
        if 'sma_20' not in df.columns or 'sma_50' not in df.columns:
            return "neutral"
            
        latest = df.iloc[-1]
        sma20 = latest['sma_20']
        sma50 = latest['sma_50']
        sma200 = latest.get('sma_200', None)
        close = latest['close']
        
        # Enhanced trend detection
        # Check for stronger trends with multiple confirmations
        strong_uptrend = (close > sma20 > sma50 and 
                        (sma200 is None or sma50 > sma200) and
                        df['close'].pct_change(10).iloc[-1] > 0)
                        
        strong_downtrend = (close < sma20 < sma50 and
                          (sma200 is None or sma50 < sma200) and
                          df['close'].pct_change(10).iloc[-1] < 0)
                          
        # Check 20MA slope for trend strength
        sma20_slope = df['sma_20'].diff(5).iloc[-1] / df['sma_20'].iloc[-6] * 100  # % change over 5 periods
        
        if strong_uptrend and sma20_slope > 0:
            return "Long"
        elif strong_downtrend and sma20_slope < 0:
            return "Short"
        
        # Moderate trend
        if close > sma20 > sma50:
            return "Long"
        elif close < sma20 < sma50:
            return "Short"
        
        # Weaker trend signals
        if close > sma20 and close > sma50:
            return "Long"
        elif close < sma20 and close < sma50:
            return "Short"
            
        return "neutral"
    
    def _check_rsi(self, df_15m, df_1h):
        """
        Check for RSI signals on multiple timeframes
        
        Args:
            df_15m: 15m data with RSI
            df_1h: 1h data with RSI
            
        Returns:
            Signal direction or "neutral"
        """
        if 'rsi' not in df_15m.columns or 'rsi' not in df_1h.columns:
            return "neutral"
            
        rsi_15m = df_15m['rsi'].iloc[-1]
        rsi_1h = df_1h['rsi'].iloc[-1]
        
        # Check for extremely overbought/oversold conditions
        if rsi_15m < 30 and rsi_1h < 35:
            return "Long"  # Oversold on both timeframes
        elif rsi_15m > 70 and rsi_1h > 65:
            return "Short"  # Overbought on both timeframes
            
        # Check for RSI divergence
        price_change_15m = df_15m['close'].pct_change(5).iloc[-1] * 100
        price_change_1h = df_1h['close'].pct_change(5).iloc[-1] * 100
        
        # Bullish divergence: price making lower lows but RSI making higher lows
        if price_change_15m < -1 and df_15m['rsi'].diff(3).iloc[-1] > 0:
            return "Long"
            
        # Bearish divergence: price making higher highs but RSI making lower highs
        if price_change_15m > 1 and df_15m['rsi'].diff(3).iloc[-1] < 0:
            return "Short"
            
        # No clear signal
        return "neutral"
    
    def _check_bollinger_bands(self, df_15m, df_1h):
        """
        Check for Bollinger Bands setups
        
        Args:
            df_15m: 15m data with Bollinger Bands
            df_1h: 1h data with Bollinger Bands
            
        Returns:
            Signal direction or "neutral"
        """
        # Check if Bollinger Bands are present
        bb_columns = ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
        if not all(col in df_15m.columns for col in bb_columns):
            return "neutral"
            
        latest_15m = df_15m.iloc[-1]
        close_15m = latest_15m['close']
        upper_15m = latest_15m['BBU_20_2.0']
        lower_15m = latest_15m['BBL_20_2.0']
        middle_15m = latest_15m['BBM_20_2.0']
        
        # Calculate Bollinger Band Width (volatility indicator)
        bb_width = (upper_15m - lower_15m) / middle_15m
        
        # Check for price near or outside the bands
        # Price near lower band (potential long)
        if close_15m <= lower_15m * 1.02:
            # Check if BB width is expanding (increasing volatility after contraction)
            if bb_width > df_15m['BBU_20_2.0'].sub(df_15m['BBL_20_2.0']).div(df_15m['BBM_20_2.0']).rolling(10).mean().iloc[-1]:
                return "Long"
                
        # Price near upper band (potential short)
        if close_15m >= upper_15m * 0.98:
            # Check if BB width is expanding (increasing volatility after contraction)
            if bb_width > df_15m['BBU_20_2.0'].sub(df_15m['BBL_20_2.0']).div(df_15m['BBM_20_2.0']).rolling(10).mean().iloc[-1]:
                return "Short"
                
        # Check for Bollinger Band squeeze (low volatility period)
        recent_bb_width = df_15m['BBU_20_2.0'].sub(df_15m['BBL_20_2.0']).div(df_15m['BBM_20_2.0']).rolling(20).mean()
        
        if bb_width < recent_bb_width.quantile(0.2).iloc[-1]:
            # Squeeze detected, look for breakout direction
            if df_15m['close'].diff(3).iloc[-1] > 0 and df_15m['volume'].pct_change(3).iloc[-1] > 0.1:
                return "Long"  # Upward breakout with volume
            elif df_15m['close'].diff(3).iloc[-1] < 0 and df_15m['volume'].pct_change(3).iloc[-1] > 0.1:
                return "Short"  # Downward breakout with volume
        
        return "neutral"
    
    def _determine_trade_direction(self, overall_trend, trend_strength, rsi_signal, bb_signal,
                               market_sentiment, forecast_direction):
        """
        Determine final trade direction based on multiple factors
        
        Args:
            overall_trend: Trend direction from moving averages
            trend_strength: Strength of the trend
            rsi_signal: Signal from RSI analysis
            bb_signal: Signal from Bollinger Bands analysis
            market_sentiment: Sentiment from news analysis
            forecast_direction: Direction from price prediction model
            
        Returns:
            Final trade direction: "Long", "Short", or "neutral"
        """
        # Count signals in each direction
        long_signals = 0
        short_signals = 0
        
        # Add weights to different signals
        if overall_trend == "Long":
            if trend_strength == "strong":
                long_signals += 3
            elif trend_strength == "moderate":
                long_signals += 2
            else:
                long_signals += 1
        elif overall_trend == "Short":
            if trend_strength == "strong":
                short_signals += 3
            elif trend_strength == "moderate":
                short_signals += 2
            else:
                short_signals += 1
                
        if rsi_signal == "Long":
            long_signals += 2
        elif rsi_signal == "Short":
            short_signals += 2
            
        if bb_signal == "Long":
            long_signals += 2
        elif bb_signal == "Short":
            short_signals += 2
            
        if market_sentiment == "bullish":
            long_signals += 1
        elif market_sentiment == "bearish":
            short_signals += 1
            
        if forecast_direction == "bullish":
            long_signals += 1
        elif forecast_direction == "bearish":
            short_signals += 1
            
        # Consider market conditions - for cryptocurrencies, we need strong confirmation
        # Require at least 5 points of signal strength in either direction
        min_signal_threshold = 5
            
        # Determine direction based on weighted signals
        if long_signals >= min_signal_threshold and long_signals > short_signals + 1:
            return "Long"
        elif short_signals >= min_signal_threshold and short_signals > long_signals + 1:
            return "Short"
        else:
            return "neutral"
            
    def _calculate_confidence(self, trade_direction, overall_trend, trend_strength, 
                        rsi_signal, bb_signal, market_sentiment, sentiment_score,
                        forecast_direction, forecast_confidence, risk_reward):
        """
        Calculate confidence level for the trade (0-100%)
        
        Returns:
            Confidence score (0-100)
        """
        if trade_direction == "neutral":
            return 0
            
        # Base confidence
        confidence = 50
        
        # Add or subtract based on alignment of signals
        # Trend alignment
        if overall_trend == trade_direction:
            if trend_strength == "strong":
                confidence += 15
            elif trend_strength == "moderate":
                confidence += 10
            else:
                confidence += 5
        elif overall_trend != "neutral":
            confidence -= 10
            
        # RSI alignment
        if rsi_signal == trade_direction:
            confidence += 8
        elif rsi_signal != "neutral":
            confidence -= 5
            
        # Bollinger Bands alignment
        if bb_signal == trade_direction:
            confidence += 8
        elif bb_signal != "neutral":
            confidence -= 5
            
        # Sentiment alignment
        if (market_sentiment == "bullish" and trade_direction == "Long") or \
           (market_sentiment == "bearish" and trade_direction == "Short"):
            confidence += 5 + min(5, abs(sentiment_score) * 5)
        elif market_sentiment != "neutral":
            confidence -= 5
            
        # Forecast alignment
        if (forecast_direction == "bullish" and trade_direction == "Long") or \
           (forecast_direction == "bearish" and trade_direction == "Short"):
            confidence += forecast_confidence // 10  # Add 0-10 based on forecast confidence
        elif forecast_direction != "neutral":
            confidence -= 5
            
        # Risk/reward factor
        if risk_reward:
            if risk_reward >= 3:
                confidence += 10
            elif risk_reward >= 2:
                confidence += 5
            elif risk_reward < 1:
                confidence -= 10
                
        # Ensure confidence is in range 0-100
        confidence = max(0, min(100, confidence))
        
        return confidence
    
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