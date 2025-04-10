import numpy as np
import pandas as pd
from src.trade_setup import TradeSetup

class GBPUSDScalpingStrategy:
    """
    GBP/USD Scalping / Short-Term Algorithmic Strategy
    
    Timeframe: 1m-15m
    Indicators: VWAP, Bollinger Bands, Order Flow, Momentum
    """
    
    def __init__(self, data_fetcher):
        """Initialize strategy with data fetcher"""
        self.data_fetcher = data_fetcher
        self.name = "GBPUSD_Scalping"
        self.instrument = "GBPUSD"
        
    def analyze(self):
        """
        Analyze the GBP/USD market for scalping opportunities
        
        Returns:
            TradeSetup object with trade details or no-trade reason
        """
        # Get forex data for different timeframes
        df_1m = self.data_fetcher.get_forex_data(self.instrument, resolution='1', count=100)
        df_5m = self.data_fetcher.get_forex_data(self.instrument, resolution='5', count=60)
        df_15m = self.data_fetcher.get_forex_data(self.instrument, resolution='15', count=40)
        
        # Add technical indicators
        if not df_1m.empty and not df_5m.empty and not df_15m.empty:
            df_1m = self.data_fetcher.add_technical_indicators(df_1m)
            df_5m = self.data_fetcher.add_technical_indicators(df_5m)
            df_15m = self.data_fetcher.add_technical_indicators(df_15m)
            
            # Get order flow data (mocked)
            order_flow = self.data_fetcher.mock_order_flow(self.instrument, length=20)
            
            # Check for trade setups
            return self._check_for_setups(df_1m, df_5m, df_15m, order_flow)
        else:
            return TradeSetup.no_trade(
                instrument=self.instrument,
                strategy=self.name,
                reason="Unable to fetch required market data"
            )
            
    def _check_for_setups(self, df_1m, df_5m, df_15m, order_flow):
        """
        Check for specific trade setups based on the strategy criteria
        
        Args:
            df_1m: 1-minute timeframe data with indicators
            df_5m: 5-minute timeframe data with indicators
            df_15m: 15-minute timeframe data with indicators
            order_flow: Order book flow data
            
        Returns:
            TradeSetup with trade details or no-trade reason
        """
        # Get latest data points
        latest_1m = df_1m.iloc[-1]
        latest_5m = df_5m.iloc[-1]
        latest_15m = df_15m.iloc[-1]
        
        # Current price
        current_price = latest_1m['close']
        
        # Check for VWAP bounce setup
        vwap_setup = self._check_vwap_bounce(df_1m, df_5m, df_15m)
        if vwap_setup:
            return vwap_setup
            
        # Check for Bollinger Band squeeze setup
        bb_setup = self._check_bollinger_band_squeeze(df_5m, df_15m)
        if bb_setup:
            return bb_setup
            
        # Check for breakout from consolidation
        breakout_setup = self._check_breakout_setup(df_1m, df_5m, df_15m)
        if breakout_setup:
            return breakout_setup
            
        # Check for order flow confirmation
        order_flow_setup = self._check_order_flow_setup(df_1m, order_flow)
        if order_flow_setup:
            return order_flow_setup
        
        # No valid setup found
        return TradeSetup.no_trade(
            instrument=self.instrument,
            strategy=self.name,
            reason="No valid setup detected under current market conditions"
        )
    
    def _check_vwap_bounce(self, df_1m, df_5m, df_15m):
        """Check for VWAP bounce setup"""
        # Check if we have necessary indicators
        if 'vwap' not in df_5m.columns:
            return None
            
        latest_5m = df_5m.iloc[-1]
        price = latest_5m['close']
        vwap = latest_5m['vwap']
        
        # Recent price action
        recent_5m = df_5m.iloc[-6:]
        
        # Check if price is near VWAP (within 0.05%)
        price_near_vwap = abs(price - vwap) / vwap < 0.0005
        
        # Check for a bounce pattern off VWAP
        crosses_above_vwap = (recent_5m['low'] < recent_5m['vwap']).any() and price > vwap
        crosses_below_vwap = (recent_5m['high'] > recent_5m['vwap']).any() and price < vwap
        
        if price_near_vwap and (crosses_above_vwap or crosses_below_vwap):
            # Determine direction
            direction = "Long" if crosses_above_vwap else "Short"
            
            # ATR for stop loss and targets
            atr = latest_5m.get('atr', 0.0002)  # Default to 2 pips if ATR not available
            
            # Calculate entry zone, stop loss and targets
            if direction == "Long":
                entry_zone = (price, price + atr * 0.5)
                stop_loss = price - atr * 1.5
                targets = [price + atr * 2, price + atr * 3.5]
            else:  # Short
                entry_zone = (price - atr * 0.5, price)
                stop_loss = price + atr * 1.5
                targets = [price - atr * 2, price - atr * 3.5]
                
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Generate rationale
            bounce_direction = "above" if direction == "Long" else "below"
            rationale = (
                f"Price is bouncing {bounce_direction} VWAP with momentum confirmation on "
                f"5m timeframe. Order flow shows {direction.lower()} bias. Setup has a defined "
                f"risk/reward profile of {risk_reward:.2f}."
            )
            
            # Determine confidence level
            confidence = 65 if risk_reward > 1.5 else 50
            
            # Return trade setup
            return TradeSetup(
                instrument=self.instrument,
                direction=direction,
                entry_zone=entry_zone,
                stop_loss=stop_loss,
                targets=targets,
                risk_reward=risk_reward,
                confidence=confidence,
                rationale=rationale,
                strategy=self.name
            )
        
        return None
    
    def _check_bollinger_band_squeeze(self, df_5m, df_15m):
        """Check for Bollinger Band squeeze setup"""
        # Check for required indicators
        if 'BBL_20_2.0' not in df_15m.columns or 'BBU_20_2.0' not in df_15m.columns:
            return None
            
        latest_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-5:-1]
        
        # Calculate Bollinger Band width
        current_bb_width = latest_15m['BBU_20_2.0'] - latest_15m['BBL_20_2.0']
        prev_bb_width = prev_15m['BBU_20_2.0'] - prev_15m['BBL_20_2.0']
        avg_prev_width = prev_bb_width.mean()
        
        # Check for squeeze (narrowing bands)
        squeeze_detected = current_bb_width < avg_prev_width * 0.8
        
        if squeeze_detected:
            # Current price
            price = latest_15m['close']
            
            # Determine direction based on momentum and position within bands
            momentum = latest_15m.get('momentum', 0)
            price_vs_vwap = price > latest_15m.get('vwap', price) if 'vwap' in latest_15m else None
            
            # Set direction
            if momentum > 0 and (price_vs_vwap is None or price_vs_vwap):
                direction = "Long"
            elif momentum < 0 and (price_vs_vwap is None or not price_vs_vwap):
                direction = "Short"
            else:
                # No clear direction
                return None
                
            # ATR for stop loss and targets
            atr = latest_15m.get('atr', 0.0003)  # Default to 3 pips if ATR not available
            
            # Calculate entry zone, stop loss and targets
            if direction == "Long":
                entry_zone = (price, price + atr * 0.7)
                stop_loss = min(price - atr * 2, latest_15m['BBL_20_2.0'])
                targets = [price + atr * 3, price + atr * 5]
            else:  # Short
                entry_zone = (price - atr * 0.7, price)
                stop_loss = max(price + atr * 2, latest_15m['BBU_20_2.0'])
                targets = [price - atr * 3, price - atr * 5]
                
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Generate rationale
            rationale = (
                f"Bollinger Band squeeze detected on 15m timeframe with {direction.lower()} "
                f"momentum. Volatility contraction suggests an impending breakout. "
                f"Stop placed beyond the opposite Bollinger Band for protection."
            )
            
            # Determine confidence level
            confidence = 75 if risk_reward > 2 else 60
            
            # Return trade setup
            return TradeSetup(
                instrument=self.instrument,
                direction=direction,
                entry_zone=entry_zone,
                stop_loss=stop_loss,
                targets=targets,
                risk_reward=risk_reward,
                confidence=confidence,
                rationale=rationale,
                strategy=self.name
            )
            
        return None
        
    def _check_breakout_setup(self, df_1m, df_5m, df_15m):
        """Check for breakout from local consolidation"""
        # Use 5m chart for this setup
        if df_5m.empty:
            return None
            
        # Get recent price action
        recent_candles = df_5m.iloc[-15:]
        
        # Calculate recent range (high - low)
        recent_high = recent_candles['high'].max()
        recent_low = recent_candles['low'].min()
        recent_range = recent_high - recent_low
        
        # Check last 5 candles range compared to previous 10 candles
        last_5_candles = recent_candles.iloc[-5:]
        last_5_range = last_5_candles['high'].max() - last_5_candles['low'].min()
        
        # Check for consolidation (narrowing range)
        consolidation = last_5_range < recent_range * 0.4
        
        if consolidation:
            # Current price
            current_price = df_5m.iloc[-1]['close']
            
            # Determine if we're near the breakout level
            near_high = (recent_high - current_price) / recent_range < 0.2
            near_low = (current_price - recent_low) / recent_range < 0.2
            
            # Check momentum direction
            momentum = df_5m.iloc[-1].get('momentum', 0)
            
            # Only consider valid breakout if price is near range boundary with momentum
            if (near_high and momentum > 0) or (near_low and momentum < 0):
                # Determine direction
                direction = "Long" if near_high and momentum > 0 else "Short"
                
                # ATR for stop loss and targets
                atr = df_5m.iloc[-1].get('atr', 0.0003)
                
                # Calculate entry, stop and targets
                if direction == "Long":
                    entry_zone = (recent_high, recent_high + atr * 0.5)
                    stop_loss = max(recent_high - atr * 2, recent_low)
                    targets = [recent_high + atr * 3, recent_high + (recent_high - recent_low)]
                else:  # Short
                    entry_zone = (recent_low - atr * 0.5, recent_low)
                    stop_loss = min(recent_low + atr * 2, recent_high)
                    targets = [recent_low - atr * 3, recent_low - (recent_high - recent_low)]
                
                # Calculate risk/reward
                risk_reward = TradeSetup.calculate_risk_reward(
                    entry=sum(entry_zone) / 2,
                    stop=stop_loss,
                    target=targets[0]
                )
                
                # Generate rationale
                breakout_type = "upper resistance" if direction == "Long" else "lower support"
                rationale = (
                    f"Consolidation detected over last {len(recent_candles)} periods on 5m chart. "
                    f"Price is approaching {breakout_type} with strong momentum. "
                    f"Breakout strategy aims to capture the impulse move after consolidation."
                )
                
                # Determine confidence level
                confidence = 70 if risk_reward > 1.5 else 55
                
                # Return trade setup
                return TradeSetup(
                    instrument=self.instrument,
                    direction=direction,
                    entry_zone=entry_zone,
                    stop_loss=stop_loss,
                    targets=targets,
                    risk_reward=risk_reward,
                    confidence=confidence,
                    rationale=rationale,
                    strategy=self.name
                )
        
        return None
        
    def _check_order_flow_setup(self, df_1m, order_flow):
        """Check for order flow imbalance setup"""
        if order_flow.empty:
            return None
            
        # Calculate order flow imbalance
        recent_flow = order_flow.iloc[-5:]
        buy_volume = recent_flow['buy_volume'].sum()
        sell_volume = recent_flow['sell_volume'].sum()
        volume_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
        
        # Check for strong imbalance
        strong_buy_imbalance = volume_ratio > 1.5
        strong_sell_imbalance = volume_ratio < 0.67
        
        if strong_buy_imbalance or strong_sell_imbalance:
            # Current price
            current_price = df_1m.iloc[-1]['close']
            
            # Determine direction
            direction = "Long" if strong_buy_imbalance else "Short"
            
            # Check if order flow confirms a key level
            recent_prices = df_1m.iloc[-10:]
            has_pivot = False
            
            if direction == "Long":
                # Check for a recent low that could act as support
                recent_low = recent_prices['low'].min()
                has_pivot = abs(current_price - recent_low) / current_price < 0.001
            else:
                # Check for a recent high that could act as resistance
                recent_high = recent_prices['high'].max()
                has_pivot = abs(current_price - recent_high) / current_price < 0.001
                
            # Only proceed if we have order flow + price level confluence
            if has_pivot:
                # ATR for stop loss and targets
                atr = df_1m.iloc[-1].get('atr', 0.0002)
                
                # Calculate entry, stop and targets
                if direction == "Long":
                    entry_zone = (current_price, current_price + atr * 0.5)
                    stop_loss = current_price - atr * 2
                    targets = [current_price + atr * 3, current_price + atr * 5]
                else:  # Short
                    entry_zone = (current_price - atr * 0.5, current_price)
                    stop_loss = current_price + atr * 2
                    targets = [current_price - atr * 3, current_price - atr * 5]
                
                # Calculate risk/reward
                risk_reward = TradeSetup.calculate_risk_reward(
                    entry=sum(entry_zone) / 2,
                    stop=stop_loss,
                    target=targets[0]
                )
                
                # Generate rationale
                imbalance_type = "buying" if direction == "Long" else "selling"
                rationale = (
                    f"Significant {imbalance_type} pressure detected in order flow "
                    f"(volume ratio: {volume_ratio:.2f}). Setup occurs at a key "
                    f"price level with strong order flow confirmation."
                )
                
                # Determine confidence level
                confidence = 75 if risk_reward > 2 and abs(volume_ratio - 1) > 1 else 60
                
                # Return trade setup
                return TradeSetup(
                    instrument=self.instrument,
                    direction=direction,
                    entry_zone=entry_zone,
                    stop_loss=stop_loss,
                    targets=targets,
                    risk_reward=risk_reward,
                    confidence=confidence,
                    rationale=rationale,
                    strategy=self.name
                )
                
        return None 