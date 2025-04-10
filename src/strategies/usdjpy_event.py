import numpy as np
import pandas as pd
import datetime
from src.trade_setup import TradeSetup

class USDJPYEventStrategy:
    """
    JPY (USD/JPY) – Event-Driven Macro + Volatility Strategy with SMC (Smart Money Concepts)
    
    Inputs: Central bank statements, CPI/NFP data, yield curve,
    VIX correlations, fair value gaps, liquidity, and market structure
    """
    
    def __init__(self, data_fetcher):
        """Initialize strategy with data fetcher"""
        self.data_fetcher = data_fetcher
        self.name = "USDJPY_Event"
        self.instrument = "USDJPY"
        
        # Important economic events that impact JPY
        self.key_events = [
            "BoJ Policy Rate", "Fed Rate Decision", "Interest Rate Decision",
            "CPI", "Non-Farm Payroll", "Unemployment Rate", "GDP",
            "Retail Sales", "Industrial Production"
        ]
        
    def analyze(self):
        """
        Analyze USD/JPY for event-driven opportunities
        
        Returns:
            TradeSetup object with trade details or no-trade reason
        """
        # Get forex data for different timeframes
        df_1h = self.data_fetcher.get_forex_data(self.instrument, resolution='60', count=100)
        df_4h = self.data_fetcher.get_forex_data(self.instrument, resolution='240', count=50)
        df_daily = self.data_fetcher.get_forex_data(self.instrument, resolution='D', count=50)
        
        # Add technical indicators
        if not df_1h.empty and not df_4h.empty and not df_daily.empty:
            df_1h = self.data_fetcher.add_technical_indicators(df_1h)
            df_4h = self.data_fetcher.add_technical_indicators(df_4h)
            df_daily = self.data_fetcher.add_technical_indicators(df_daily)
            
            # Get VIX data
            vix_data = self.data_fetcher.get_vix_data(days=30)
            
            # Get economic calendar events
            calendar = self.data_fetcher.get_economic_calendar()
            
            # Get yield curve data
            yield_data = self.data_fetcher.get_yield_curve_data()
            
            # Get market news
            news = self.data_fetcher.get_news(category="forex", min_id=0)
            
            # Check for trade setups based on event-driven approach
            return self._check_for_setups(df_1h, df_4h, df_daily, vix_data, calendar, yield_data, news)
        else:
            return TradeSetup.no_trade(
                instrument=self.instrument,
                strategy=self.name,
                reason="Unable to fetch required market data"
            )
    
    def _check_for_setups(self, df_1h, df_4h, df_daily, vix_data, calendar, yield_data, news):
        """
        Check for event-driven trade setups
        
        Returns:
            TradeSetup with trade details or no-trade reason
        """
        # Check for fair value gap setup (highest priority)
        fvg_setup = self._check_fair_value_gap(df_1h, df_4h)
        if fvg_setup:
            return fvg_setup
            
        # Check for market structure break and continuation
        structure_setup = self._check_market_structure(df_1h, df_4h, df_daily)
        if structure_setup:
            return structure_setup
            
        # Check for upcoming economic events
        event_setup = self._check_upcoming_events(df_1h, calendar)
        if event_setup:
            return event_setup
            
        # Check for yield curve based setup
        yield_setup = self._check_yield_curve_setup(df_1h, yield_data)
        if yield_setup:
            return yield_setup
            
        # Check for VIX correlation setup
        vix_setup = self._check_vix_correlation(df_daily, vix_data)
        if vix_setup:
            return vix_setup
            
        # Check for news-based setup
        news_setup = self._check_news_setup(df_1h, news)
        if news_setup:
            return news_setup
            
        # No valid setup found
        return TradeSetup.no_trade(
            instrument=self.instrument,
            strategy=self.name,
            reason="No significant macro events or setups detected for JPY"
        )
    
    def _check_fair_value_gap(self, df_1h, df_4h):
        """
        Check for Fair Value Gap (FVG) setups
        
        Fair Value Gaps occur when there's a gap between candle bodies that remains unfilled,
        indicating a strong imbalance that often gets filled later
        """
        # Need enough data for this setup
        if len(df_1h) < 24:
            return None
            
        # Look for FVGs on 1 hour timeframe
        bullish_fvgs = []  # (bottom of gap, top of gap, age in candles)
        bearish_fvgs = []  # (bottom of gap, top of gap, age in candles)
        
        # Scan for Fair Value Gaps (skip the most recent 3 candles)
        for i in range(3, len(df_1h) - 3):
            # Bullish Fair Value Gap: Current candle low is above previous candle high
            if df_1h.iloc[i]['low'] > df_1h.iloc[i-2]['high']:
                # Gap between previous high and current low
                fvg_bottom = df_1h.iloc[i-2]['high']
                fvg_top = df_1h.iloc[i]['low']
                
                # Check if FVG is still valid (not filled yet)
                is_filled = False
                for j in range(i+1, len(df_1h)):
                    if df_1h.iloc[j]['low'] <= fvg_top:
                        is_filled = True
                        break
                        
                if not is_filled:
                    # Record this FVG with its age (most recent gets priority)
                    age = len(df_1h) - i
                    bullish_fvgs.append((fvg_bottom, fvg_top, age))
            
            # Bearish Fair Value Gap: Current candle high is below previous candle low
            if df_1h.iloc[i]['high'] < df_1h.iloc[i-2]['low']:
                # Gap between current high and previous low
                fvg_bottom = df_1h.iloc[i]['high']
                fvg_top = df_1h.iloc[i-2]['low']
                
                # Check if FVG is still valid (not filled yet)
                is_filled = False
                for j in range(i+1, len(df_1h)):
                    if df_1h.iloc[j]['high'] >= fvg_bottom:
                        is_filled = True
                        break
                        
                if not is_filled:
                    # Record this FVG with its age (most recent gets priority)
                    age = len(df_1h) - i
                    bearish_fvgs.append((fvg_bottom, fvg_top, age))
            
        # If no FVGs found, check on 4h timeframe
        if not bullish_fvgs and not bearish_fvgs and len(df_4h) >= 20:
            for i in range(3, len(df_4h) - 3):
                # Bullish Fair Value Gap
                if df_4h.iloc[i]['low'] > df_4h.iloc[i-2]['high']:
                    fvg_bottom = df_4h.iloc[i-2]['high']
                    fvg_top = df_4h.iloc[i]['low']
                    
                    is_filled = False
                    for j in range(i+1, len(df_4h)):
                        if df_4h.iloc[j]['low'] <= fvg_top:
                            is_filled = True
                            break
                            
                    if not is_filled:
                        age = len(df_4h) - i
                        bullish_fvgs.append((fvg_bottom, fvg_top, age))
                
                # Bearish Fair Value Gap
                if df_4h.iloc[i]['high'] < df_4h.iloc[i-2]['low']:
                    fvg_bottom = df_4h.iloc[i]['high']
                    fvg_top = df_4h.iloc[i-2]['low']
                    
                    is_filled = False
                    for j in range(i+1, len(df_4h)):
                        if df_4h.iloc[j]['high'] >= fvg_bottom:
                            is_filled = True
                            break
                            
                    if not is_filled:
                        age = len(df_4h) - i
                        bearish_fvgs.append((fvg_bottom, fvg_top, age))
        
        # Current price and market state
        current_price = df_1h.iloc[-1]['close']
        
        # Analyze 4h trend
        trend = "neutral"
        recent_4h = df_4h.iloc[-10:]
        
        if recent_4h['close'].iloc[-1] > recent_4h['close'].iloc[0] * 1.005:
            trend = "bullish"
        elif recent_4h['close'].iloc[-1] < recent_4h['close'].iloc[0] * 0.995:
            trend = "bearish"
            
        # Sort FVGs by age (most recent first)
        bullish_fvgs.sort(key=lambda x: x[2])
        bearish_fvgs.sort(key=lambda x: x[2])
        
        # Check for price approaching a FVG
        setup = None
        
        # Look for price approaching a bearish FVG from below (for short trades in bearish trend)
        if trend == "bearish" and bearish_fvgs:
            for fvg_bottom, fvg_top, _ in bearish_fvgs:
                # Is price near the bottom of the bearish FVG?
                if current_price >= fvg_bottom * 0.997 and current_price <= fvg_bottom * 1.003:
                    # ATR for stop and target calculations
                    atr = df_1h.iloc[-1].get('atr', 0.1)
                    
                    # Calculate entry, stop and targets
                    entry_zone = (fvg_bottom * 0.999, fvg_bottom * 1.001)
                    stop_loss = fvg_top + atr * 0.5
                    
                    # First target is the middle of the FVG
                    mid_fvg = (fvg_top + fvg_bottom) / 2
                    # Second target is the bottom of the FVG plus the height of the FVG
                    fvg_height = fvg_top - fvg_bottom
                    
                    targets = [mid_fvg, fvg_bottom - fvg_height]
                    
                    # Calculate risk/reward
                    risk_reward = TradeSetup.calculate_risk_reward(
                        entry=sum(entry_zone) / 2,
                        stop=stop_loss,
                        target=targets[0]
                    )
                    
                    # Only proceed if risk/reward is reasonable
                    if risk_reward >= 1.5:
                        # Generate rationale
                        rationale = (
                            f"Price approaching bearish Fair Value Gap at {fvg_bottom:.2f}-{fvg_top:.2f} in a {trend} trend. "
                            f"FVGs represent areas of imbalance that tend to get filled. Entry at the bottom of "
                            f"the FVG with confirmation of rejection. First target at the middle of the FVG, "
                            f"second target is a full extension. Risk/reward: {risk_reward:.2f}."
                        )
                        
                        # Determine confidence
                        confidence = 75 if trend == "bearish" else 65
                        
                        # Return the trade setup
                        setup = TradeSetup(
                            instrument=self.instrument,
                            direction="Short",
                            entry_zone=entry_zone,
                            stop_loss=stop_loss,
                            targets=targets,
                            risk_reward=risk_reward,
                            confidence=confidence,
                            rationale=rationale,
                            strategy=f"{self.name}_FVG"
                        )
                        break
        
        # Look for price approaching a bullish FVG from above (for long trades in bullish trend)
        elif trend == "bullish" and bullish_fvgs:
            for fvg_bottom, fvg_top, _ in bullish_fvgs:
                # Is price near the top of the bullish FVG?
                if current_price >= fvg_top * 0.997 and current_price <= fvg_top * 1.003:
                    # ATR for stop and target calculations
                    atr = df_1h.iloc[-1].get('atr', 0.1)
                    
                    # Calculate entry, stop and targets
                    entry_zone = (fvg_top * 0.999, fvg_top * 1.001)
                    stop_loss = fvg_bottom - atr * 0.5
                    
                    # First target is the middle of the FVG
                    mid_fvg = (fvg_top + fvg_bottom) / 2
                    # Second target is the top of the FVG plus the height of the FVG
                    fvg_height = fvg_top - fvg_bottom
                    
                    targets = [mid_fvg, fvg_top + fvg_height]
                    
                    # Calculate risk/reward
                    risk_reward = TradeSetup.calculate_risk_reward(
                        entry=sum(entry_zone) / 2,
                        stop=stop_loss,
                        target=targets[0]
                    )
                    
                    # Only proceed if risk/reward is reasonable
                    if risk_reward >= 1.5:
                        # Generate rationale
                        rationale = (
                            f"Price approaching bullish Fair Value Gap at {fvg_bottom:.2f}-{fvg_top:.2f} in a {trend} trend. "
                            f"FVGs represent areas of imbalance that tend to get filled. Entry at the top of "
                            f"the FVG with confirmation of support. First target at the middle of the FVG, "
                            f"second target is a full extension. Risk/reward: {risk_reward:.2f}."
                        )
                        
                        # Determine confidence
                        confidence = 75 if trend == "bullish" else 65
                        
                        # Return the trade setup
                        setup = TradeSetup(
                            instrument=self.instrument,
                            direction="Long",
                            entry_zone=entry_zone,
                            stop_loss=stop_loss,
                            targets=targets,
                            risk_reward=risk_reward,
                            confidence=confidence,
                            rationale=rationale,
                            strategy=f"{self.name}_FVG"
                        )
                        break
                        
        return setup
        
    def _check_market_structure(self, df_1h, df_4h, df_daily):
        """
        Analyze market structure for trends, shifts, and continuation patterns
        
        Identifies higher highs/lows in bullish markets and lower highs/lows in bearish markets
        Detects structure breaks and potential reversals
        """
        # Need enough data
        if len(df_4h) < 30:
            return None
            
        # Get current price
        current_price = df_4h.iloc[-1]['close']
        
        # Identify swing points on 4h timeframe
        highs = []
        lows = []
        
        # Find last 6 significant highs and lows, skipping the most recent 3 candles
        for i in range(5, len(df_4h) - 3):
            # Significant high (higher than 2 candles before and after)
            if (df_4h.iloc[i]['high'] > df_4h.iloc[i-1]['high'] and 
                df_4h.iloc[i]['high'] > df_4h.iloc[i-2]['high'] and
                df_4h.iloc[i]['high'] > df_4h.iloc[i+1]['high'] and
                df_4h.iloc[i]['high'] > df_4h.iloc[i+2]['high']):
                highs.append((i, df_4h.iloc[i]['high']))
                if len(highs) >= 6:
                    highs.pop(0)  # Keep only the 6 most recent
                    
            # Significant low (lower than 2 candles before and after)
            if (df_4h.iloc[i]['low'] < df_4h.iloc[i-1]['low'] and 
                df_4h.iloc[i]['low'] < df_4h.iloc[i-2]['low'] and
                df_4h.iloc[i]['low'] < df_4h.iloc[i+1]['low'] and
                df_4h.iloc[i]['low'] < df_4h.iloc[i+2]['low']):
                lows.append((i, df_4h.iloc[i]['low']))
                if len(lows) >= 6:
                    lows.pop(0)  # Keep only the 6 most recent
        
        # Need at least 4 swing points to analyze structure
        if len(highs) < 4 or len(lows) < 4:
            return None
            
        # Sort swing points by index
        highs.sort(key=lambda x: x[0])
        lows.sort(key=lambda x: x[0])
        
        # Check for higher highs and higher lows (uptrend)
        is_uptrend = (highs[-1][1] > highs[-2][1] > highs[-3][1] and 
                      lows[-1][1] > lows[-2][1] > lows[-3][1])
                      
        # Check for lower highs and lower lows (downtrend)
        is_downtrend = (highs[-1][1] < highs[-2][1] < highs[-3][1] and 
                       lows[-1][1] < lows[-2][1] < lows[-3][1])
        
        # Check for a potential structure break
        structure_broken = False
        structure_direction = None
        
        # Bullish structure break (break of a previous higher low)
        if (not is_uptrend and not is_downtrend and 
            current_price > lows[-2][1] and 
            lows[-1][1] < lows[-2][1]):
            structure_broken = True
            structure_direction = "bullish"
            
        # Bearish structure break (break of a previous higher high)
        if (not is_uptrend and not is_downtrend and 
            current_price < highs[-2][1] and 
            highs[-1][1] > highs[-2][1]):
            structure_broken = True
            structure_direction = "bearish"
            
        # Look for a tradable setup based on the structure
        if is_uptrend or (structure_broken and structure_direction == "bullish"):
            # Identify a recent pullback in an uptrend
            recent_low = min(df_4h.iloc[-5:]['low'])
            recent_low_idx = df_4h.iloc[-5:]['low'].idxmin()
            
            # If price has pulled back to a reasonable level but still respects structure
            if (current_price > recent_low * 1.002 and 
                current_price < recent_low * 1.01 and
                recent_low > lows[-2][1]):
                
                # ATR for stop and target calculations
                atr = df_4h.iloc[-1].get('atr', 0.1)
                
                # Calculate entry, stop and targets
                entry_zone = (current_price, current_price + atr * 0.3)
                stop_loss = min(recent_low - atr * 0.5, lows[-2][1] - atr * 0.2)
                
                # First target is the next structure high
                target1 = highs[-1][1]
                # Second target is an extension based on the range
                target2 = target1 + (target1 - stop_loss)
                
                targets = [target1, target2]
                
                # Calculate risk/reward
                risk_reward = TradeSetup.calculate_risk_reward(
                    entry=sum(entry_zone) / 2,
                    stop=stop_loss,
                    target=targets[0]
                )
                
                # Only proceed if risk/reward is reasonable
                if risk_reward >= 1.5:
                    # Generate rationale based on structure type
                    rationale = ""
                    if is_uptrend:
                        rationale = (
                            f"Strong uptrend structure with higher highs and higher lows. "
                            f"Price has pulled back to {recent_low:.2f} and now showing signs of continuation. "
                            f"Entry after pullback with first target at previous high of {target1:.2f}. "
                            f"Risk/reward: {risk_reward:.2f}."
                        )
                    else:
                        rationale = (
                            f"Bullish structure break detected. Previous structure low {lows[-2][1]:.2f} has been "
                            f"broken, signaling a potential trend change. "
                            f"Entry on first pullback with stop below recent swing low. "
                            f"Risk/reward: {risk_reward:.2f}."
                        )
                    
                    # Determine confidence level
                    confidence = 80 if is_uptrend else 70
                    
                    # Return the trade setup
                    return TradeSetup(
                        instrument=self.instrument,
                        direction="Long",
                        entry_zone=entry_zone,
                        stop_loss=stop_loss,
                        targets=targets,
                        risk_reward=risk_reward,
                        confidence=confidence,
                        rationale=rationale,
                        strategy=f"{self.name}_Structure"
                    )
                    
        elif is_downtrend or (structure_broken and structure_direction == "bearish"):
            # Identify a recent bounce in a downtrend
            recent_high = max(df_4h.iloc[-5:]['high'])
            recent_high_idx = df_4h.iloc[-5:]['high'].idxmax()
            
            # If price has bounced to a reasonable level but still respects structure
            if (current_price < recent_high * 0.998 and 
                current_price > recent_high * 0.99 and
                recent_high < highs[-2][1]):
                
                # ATR for stop and target calculations
                atr = df_4h.iloc[-1].get('atr', 0.1)
                
                # Calculate entry, stop and targets
                entry_zone = (current_price - atr * 0.3, current_price)
                stop_loss = max(recent_high + atr * 0.5, highs[-2][1] + atr * 0.2)
                
                # First target is the next structure low
                target1 = lows[-1][1]
                # Second target is an extension based on the range
                target2 = target1 - (stop_loss - target1)
                
                targets = [target1, target2]
                
                # Calculate risk/reward
                risk_reward = TradeSetup.calculate_risk_reward(
                    entry=sum(entry_zone) / 2,
                    stop=stop_loss,
                    target=targets[0]
                )
                
                # Only proceed if risk/reward is reasonable
                if risk_reward >= 1.5:
                    # Generate rationale based on structure type
                    rationale = ""
                    if is_downtrend:
                        rationale = (
                            f"Strong downtrend structure with lower highs and lower lows. "
                            f"Price has bounced to {recent_high:.2f} and now showing signs of continuation. "
                            f"Entry after bounce with first target at previous low of {target1:.2f}. "
                            f"Risk/reward: {risk_reward:.2f}."
                        )
                    else:
                        rationale = (
                            f"Bearish structure break detected. Previous structure high {highs[-2][1]:.2f} has been "
                            f"broken, signaling a potential trend change. "
                            f"Entry on first bounce with stop above recent swing high. "
                            f"Risk/reward: {risk_reward:.2f}."
                        )
                    
                    # Determine confidence level
                    confidence = 80 if is_downtrend else 70
                    
                    # Return the trade setup
                    return TradeSetup(
                        instrument=self.instrument,
                        direction="Short",
                        entry_zone=entry_zone,
                        stop_loss=stop_loss,
                        targets=targets,
                        risk_reward=risk_reward,
                        confidence=confidence,
                        rationale=rationale,
                        strategy=f"{self.name}_Structure"
                    )
                    
        return None
    
    def _check_upcoming_events(self, df_1h, calendar):
        """Check for upcoming key economic events that could impact JPY"""
        # Filter calendar for key events
        if not calendar:
            return None
            
        # Current time
        now = datetime.datetime.now()
        
        # Filter calendar for relevant events in next 48 hours
        upcoming_events = []
        for event in calendar:
            if 'event' not in event or 'time' not in event:
                continue
                
            # Check if event is key for JPY
            is_key_event = any(key in event['event'] for key in self.key_events)
            is_japan_event = 'japan' in event.get('country', '').lower()
            is_us_event = 'united states' in event.get('country', '').lower()
            
            if is_key_event and (is_japan_event or is_us_event):
                # Parse event time
                try:
                    event_time = datetime.datetime.fromtimestamp(event['time'])
                    if event_time > now and (event_time - now).total_seconds() < 48 * 3600:
                        upcoming_events.append(event)
                except:
                    continue
                    
        # If high-impact events coming up
        if upcoming_events:
            # Most important upcoming event
            upcoming_events.sort(key=lambda x: x.get('impact', 0), reverse=True)
            next_event = upcoming_events[0]
            
            # Current price
            current_price = df_1h.iloc[-1]['close']
            
            # ATR for volatility measurement
            atr = df_1h.iloc[-1].get('atr', 0.05)
            
            # Check recent price action direction
            price_trend = "neutral"
            recent_prices = df_1h.iloc[-12:]
            if recent_prices['close'].iloc[-1] > recent_prices['close'].iloc[0] * 1.002:
                price_trend = "bullish"
            elif recent_prices['close'].iloc[-1] < recent_prices['close'].iloc[0] * 0.998:
                price_trend = "bearish"
                
            # Determine expected market reaction
            # JPY typically strengthens (USDJPY falls) during risk-off events and BoJ hawkishness
            direction = None
            
            is_boj_event = 'boj' in next_event['event'].lower()
            is_rate_decision = 'rate' in next_event['event'].lower() or 'policy' in next_event['event'].lower()
            is_inflation_data = 'cpi' in next_event['event'].lower() or 'inflation' in next_event['event'].lower()
            is_employment = 'payroll' in next_event['event'].lower() or 'employment' in next_event['event'].lower()
            
            if is_boj_event:
                # BoJ events tend to drive volatility in JPY
                # Hawkish BoJ = stronger JPY (lower USDJPY)
                direction = "Short"  # Assume hawkish stance
            elif is_rate_decision and is_us_event:
                # Fed events impact USD/JPY directly
                # Hawkish Fed = stronger USD (higher USDJPY)
                direction = "Long"  # Assume hawkish stance
            elif is_inflation_data:
                # Higher inflation = expectation of tighter policy
                if is_us_event:
                    direction = "Long"  # Higher US inflation = stronger USD
                else:
                    direction = "Short"  # Higher JP inflation = stronger JPY
            elif is_employment:
                # Stronger employment = stronger economy = stronger currency
                if is_us_event:
                    direction = "Long"  # Stronger US employment = stronger USD
                else:
                    direction = "Short"  # Stronger JP employment = stronger JPY
            
            # Consider current price trend as additional factor
            confidence_adj = 0
            if direction == "Long" and price_trend == "bullish":
                confidence_adj = 10
            elif direction == "Short" and price_trend == "bearish":
                confidence_adj = 10
            elif direction == "Long" and price_trend == "bearish":
                confidence_adj = -10
                direction = "Short"  # Change direction if strong opposite trend
            elif direction == "Short" and price_trend == "bullish":
                confidence_adj = -10
                direction = "Long"  # Change direction if strong opposite trend
                
            # Only create setup if we have a direction
            if direction:
                # Calculate entry zone, stop loss and targets
                if direction == "Long":
                    entry_zone = (current_price, current_price + atr * 0.3)
                    stop_loss = current_price - atr * 1.5
                    targets = [current_price + atr * 2, current_price + atr * 4]
                else:  # Short
                    entry_zone = (current_price - atr * 0.3, current_price)
                    stop_loss = current_price + atr * 1.5
                    targets = [current_price - atr * 2, current_price - atr * 4]
                    
                # Calculate risk/reward
                risk_reward = TradeSetup.calculate_risk_reward(
                    entry=sum(entry_zone) / 2,
                    stop=stop_loss,
                    target=targets[0]
                )
                
                # Base confidence on event impact and setup quality
                base_confidence = 60
                impact_confidence = next_event.get('impact', 0) * 5  # Impact from 0-3
                confidence = min(90, base_confidence + impact_confidence + confidence_adj)
                
                # Generate rationale
                event_time_str = datetime.datetime.fromtimestamp(next_event['time']).strftime('%Y-%m-%d %H:%M')
                rationale = (
                    f"Upcoming {next_event['event']} on {event_time_str} is likely to "
                    f"impact {self.instrument}. Current market trend is {price_trend}. "
                    f"Position taken in anticipation of event volatility. "
                    f"Stop placed at {atr * 1.5:.2f} points to account for event volatility."
                )
                
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
        
    def _check_yield_curve_setup(self, df_1h, yield_data):
        """Check for yield curve based trading opportunities"""
        if not yield_data:
            return None
            
        # Current price
        current_price = df_1h.iloc[-1]['close']
        
        # Get key yield spreads
        us_jp_spread = 300  # Mock 3% spread between US and JP rates
        
        # Check for yield curve inversion as risk-off signal
        is_inverted_10y_2y = yield_data.get('10Y_2Y_spread', 0.1) < 0
        is_inverted_10y_3m = yield_data.get('10Y_3M_spread', 0.1) < 0
        
        # JPY is a classic safe-haven currency that tends to strengthen during risk-off events
        # USD/JPY tends to fall when there's risk aversion
        if is_inverted_10y_2y or is_inverted_10y_3m:
            direction = "Short"  # JPY strengthens in risk-off
            
            # ATR for volatility measurement
            atr = df_1h.iloc[-1].get('atr', 0.05)
            
            # Calculate entry zone, stop loss and targets
            entry_zone = (current_price - atr * 0.3, current_price)
            stop_loss = current_price + atr * 1.5
            targets = [current_price - atr * 2, current_price - atr * 3.5]
            
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Determine confidence level
            confidence = 70 if is_inverted_10y_2y and is_inverted_10y_3m else 60
            
            # Generate rationale
            inverted_parts = []
            if is_inverted_10y_2y:
                inverted_parts.append("10Y-2Y")
            if is_inverted_10y_3m:
                inverted_parts.append("10Y-3M")
                
            rationale = (
                f"Yield curve inversion detected in {' and '.join(inverted_parts)} spreads. "
                f"This typically signals economic uncertainty and risk-off sentiment. "
                f"JPY tends to strengthen (USDJPY falls) during risk-off periods due to "
                f"its safe-haven status. Current US-JP yield differential is approximately 300bps."
            )
            
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
            
        # Check for widening yield differential
        elif us_jp_spread > 250:
            direction = "Long"  # Higher US yields typically strengthen USD vs JPY
            
            # ATR for volatility measurement
            atr = df_1h.iloc[-1].get('atr', 0.05)
            
            # Calculate entry zone, stop loss and targets
            entry_zone = (current_price, current_price + atr * 0.3)
            stop_loss = current_price - atr * 1.5
            targets = [current_price + atr * 2, current_price + atr * 3.5]
            
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Determine confidence level
            confidence = 65
            
            # Generate rationale
            rationale = (
                f"Wide yield differential of approximately 300bps between US and Japan. "
                f"This interest rate differential typically attracts capital flows from JPY to USD, "
                f"leading to USDJPY appreciation. The carry trade remains favorable for USD."
            )
            
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
    
    def _check_vix_correlation(self, df_daily, vix_data):
        """Check for VIX correlation based trading opportunities"""
        if vix_data.empty or df_daily.empty:
            return None
            
        # Get latest VIX close and daily change
        latest_vix = vix_data.iloc[-1]
        vix_close = latest_vix['close']
        vix_change = (vix_close / vix_data.iloc[-2]['close'] - 1) * 100  # Daily % change
        
        # Current USDJPY price
        current_price = df_daily.iloc[-1]['close']
        
        # ATR for volatility measurement
        atr = df_daily.iloc[-1].get('atr', 0.5)  # Daily ATR can be larger, around 50-100 pips
        
        # Check for significant VIX movement
        if abs(vix_change) > 8:  # More than 8% move in VIX
            # JPY tends to strengthen (USDJPY falls) when VIX rises (risk-off)
            # JPY tends to weaken (USDJPY rises) when VIX falls (risk-on)
            
            direction = "Short" if vix_change > 0 else "Long"
            
            # Calculate entry zone, stop loss and targets based on VIX volatility
            vix_volatility_factor = min(3, 1 + abs(vix_change) / 10)  # Scale with VIX movement
            
            if direction == "Long":
                entry_zone = (current_price, current_price + atr * 0.2)
                stop_loss = current_price - atr * 1.5
                targets = [
                    current_price + atr * 1.5 * vix_volatility_factor, 
                    current_price + atr * 3 * vix_volatility_factor
                ]
            else:  # Short
                entry_zone = (current_price - atr * 0.2, current_price)
                stop_loss = current_price + atr * 1.5
                targets = [
                    current_price - atr * 1.5 * vix_volatility_factor, 
                    current_price - atr * 3 * vix_volatility_factor
                ]
                
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Determine confidence based on VIX movement strength
            confidence = min(85, 50 + abs(vix_change))
            
            # Generate rationale
            vix_action = "spike" if vix_change > 0 else "drop"
            risk_sentiment = "risk-off" if vix_change > 0 else "risk-on"
            
            rationale = (
                f"Significant VIX {vix_action} of {abs(vix_change):.1f}% indicates {risk_sentiment} "
                f"environment. JPY typically {'' if direction == 'Short' else 'weakens'} "
                f"{'strengthens' if direction == 'Short' else ''} during {risk_sentiment} periods. "
                f"Targets scaled by VIX volatility factor of {vix_volatility_factor:.1f}x."
            )
            
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
        
    def _check_news_setup(self, df_1h, news):
        """Check for news-based trading opportunities"""
        if not news:
            return None
            
        # Get recent relevant news (last 24 hours)
        now = datetime.datetime.now()
        recent_news = []
        
        for item in news[:20]:  # Check only most recent news
            if 'datetime' not in item:
                continue
                
            try:
                news_time = datetime.datetime.fromtimestamp(item['datetime'])
                if (now - news_time).total_seconds() < 24 * 3600:
                    # Check if news is related to JPY, BoJ, or Fed
                    headline = item.get('headline', '').lower()
                    if any(kw in headline for kw in ['yen', 'jpy', 'boj', 'japan', 'fed', 'fomc', 'powell']):
                        recent_news.append(item)
            except:
                continue
                
        # If we have significant JPY news
        if recent_news:
            # Analyze news sentiment
            bullish_keywords = ['hawkish', 'raise rates', 'hike', 'inflation', 'strong growth']
            bearish_keywords = ['dovish', 'cut rates', 'easing', 'recession', 'weak growth']
            
            boj_bullish_count = 0
            boj_bearish_count = 0
            fed_bullish_count = 0
            fed_bearish_count = 0
            
            for item in recent_news:
                headline = item.get('headline', '').lower()
                summary = item.get('summary', '').lower()
                text = headline + ' ' + summary
                
                # Check BoJ sentiment (hawkish BoJ = stronger JPY = lower USDJPY)
                if any(kw in text for kw in ['boj', 'japan', 'yen', 'jpy']):
                    boj_bullish_count += sum(kw in text for kw in bullish_keywords)
                    boj_bearish_count += sum(kw in text for kw in bearish_keywords)
                    
                # Check Fed sentiment (hawkish Fed = stronger USD = higher USDJPY)
                if any(kw in text for kw in ['fed', 'fomc', 'powell', 'us rates']):
                    fed_bullish_count += sum(kw in text for kw in bullish_keywords)
                    fed_bearish_count += sum(kw in text for kw in bearish_keywords)
            
            # Determine overall sentiment bias
            boj_bias = boj_bullish_count - boj_bearish_count
            fed_bias = fed_bullish_count - fed_bearish_count
            
            # BoJ hawkish (boj_bias > 0) = JPY stronger = USDJPY lower = Short
            # Fed hawkish (fed_bias > 0) = USD stronger = USDJPY higher = Long
            
            direction = None
            sentiment_strength = 0
            
            if boj_bias != 0 and abs(boj_bias) >= abs(fed_bias):
                direction = "Short" if boj_bias > 0 else "Long"
                sentiment_strength = abs(boj_bias)
            elif fed_bias != 0:
                direction = "Long" if fed_bias > 0 else "Short"
                sentiment_strength = abs(fed_bias)
                
            if direction and sentiment_strength >= 2:
                # Current price
                current_price = df_1h.iloc[-1]['close']
                
                # ATR for volatility measurement
                atr = df_1h.iloc[-1].get('atr', 0.05)
                
                # Scale targets based on sentiment strength
                sentiment_factor = min(2, 1 + sentiment_strength / 5)
                
                # Calculate entry zone, stop loss and targets
                if direction == "Long":
                    entry_zone = (current_price, current_price + atr * 0.3)
                    stop_loss = current_price - atr * 1.2
                    targets = [
                        current_price + atr * 2 * sentiment_factor, 
                        current_price + atr * 3.5 * sentiment_factor
                    ]
                else:  # Short
                    entry_zone = (current_price - atr * 0.3, current_price)
                    stop_loss = current_price + atr * 1.2
                    targets = [
                        current_price - atr * 2 * sentiment_factor, 
                        current_price - atr * 3.5 * sentiment_factor
                    ]
                    
                # Calculate risk/reward
                risk_reward = TradeSetup.calculate_risk_reward(
                    entry=sum(entry_zone) / 2,
                    stop=stop_loss,
                    target=targets[0]
                )
                
                # Determine confidence based on sentiment strength
                confidence = min(75, 50 + sentiment_strength * 8)
                
                # Generate rationale with specific news references
                recent_headline = recent_news[0].get('headline', 'Recent news')
                
                if boj_bias != 0 and abs(boj_bias) >= abs(fed_bias):
                    bias_source = "BoJ"
                    bias_type = "hawkish" if boj_bias > 0 else "dovish"
                else:
                    bias_source = "Fed"
                    bias_type = "hawkish" if fed_bias > 0 else "dovish"
                
                rationale = (
                    f"Recent news indicates {bias_type} {bias_source} stance: '{recent_headline}'. "
                    f"{'BoJ hawkishness typically strengthens JPY' if bias_source == 'BoJ' and bias_type == 'hawkish' else ''}"
                    f"{'BoJ dovishness typically weakens JPY' if bias_source == 'BoJ' and bias_type == 'dovish' else ''}"
                    f"{'Fed hawkishness typically strengthens USD' if bias_source == 'Fed' and bias_type == 'hawkish' else ''}"
                    f"{'Fed dovishness typically weakens USD' if bias_source == 'Fed' and bias_type == 'dovish' else ''}"
                    f" Trade targets scaled by sentiment strength factor of {sentiment_factor:.1f}x."
                )
                
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