import numpy as np
import pandas as pd
import datetime
from src.trade_setup import TradeSetup

class VIXVolatilityStrategy:
    """
    VIX / Volatility Indices – Vol Arbitrage + Long Gamma Options Strategy
    
    Inputs: VIX level, SPX IV vs RV, term structure shape, macro event proximity
    """
    
    def __init__(self, data_fetcher):
        """Initialize strategy with data fetcher"""
        self.data_fetcher = data_fetcher
        self.name = "VIX_Volatility"
        self.instrument = "VIX"
        
        # Important event types that impact volatility
        self.volatility_events = [
            "FOMC", "Non-Farm Payroll", "CPI", "Fed", "ECB",
            "GDP", "PMI", "Elections", "Testimony", "Vote"
        ]
        
        # Volatility regime thresholds
        self.low_vol_threshold = 15.0
        self.high_vol_threshold = 25.0
        self.extreme_vol_threshold = 35.0
        
    def analyze(self):
        """
        Analyze VIX for volatility trading opportunities
        
        Returns:
            TradeSetup object with trade details or no-trade reason
        """
        # Get VIX data
        vix_data = self.data_fetcher.get_vix_data(days=30)
        
        # Get economic calendar for volatility events
        calendar = self.data_fetcher.get_economic_calendar()
        
        # Get yield curve data for market stress indicators
        yield_data = self.data_fetcher.get_yield_curve_data()
        
        # Mock SPX IV and RV data (would be fetched from a real API)
        spx_implied_vol, spx_realized_vol = self._get_mock_iv_rv_data()
        
        # Mock VIX term structure data (would be fetched from a real API)
        term_structure = self._get_mock_term_structure()
        
        # Check for valid data
        if not vix_data.empty:
            # Check for trade setups
            return self._check_for_setups(
                vix_data, 
                calendar, 
                yield_data,
                spx_implied_vol, 
                spx_realized_vol,
                term_structure
            )
        else:
            return TradeSetup.no_trade(
                instrument=self.instrument,
                strategy=self.name,
                reason="Unable to fetch required VIX data"
            )
            
    def _get_mock_iv_rv_data(self):
        """
        Create mock IV and RV data for SPX
        This would be replaced with real data in production
        
        Returns:
            (float, float): SPX Implied Volatility, SPX Realized Volatility
        """
        # Mock the IV and RV with some randomization
        base_iv = 18.5  # Base level for implied volatility
        iv_variation = np.random.normal(0, 2)
        iv = max(8, base_iv + iv_variation)
        
        # Realized vol is typically within +/- 30% of IV
        rv_ratio = np.random.uniform(0.7, 1.3)
        rv = iv * rv_ratio
        
        return iv, rv
        
    def _get_mock_term_structure(self):
        """
        Create mock VIX futures term structure
        This would be replaced with real data in production
        
        Returns:
            dict: Mock term structure data
        """
        # Get a random base value for current VIX
        current_vix = np.random.uniform(15, 30)
        
        # Generate term structure with different shapes:
        # 1. Contango (upward sloping) - normal market
        # 2. Backwardation (downward sloping) - stressed market
        # 3. Humped - mixed signals
        
        shape_type = np.random.choice(['contango', 'backwardation', 'humped'], p=[0.6, 0.3, 0.1])
        
        term_structure = {
            'current': current_vix,
            'shape': shape_type,
            'months': {}
        }
        
        # Generate monthly values
        if shape_type == 'contango':
            # Upward sloping curve (normal)
            for i in range(1, 7):
                term_structure['months'][f'M{i}'] = current_vix * (1 + 0.03 * i + np.random.uniform(-0.01, 0.01))
        elif shape_type == 'backwardation':
            # Downward sloping curve (stressed market)
            for i in range(1, 7):
                term_structure['months'][f'M{i}'] = current_vix * (1 - 0.03 * i + np.random.uniform(-0.01, 0.01))
        else:  # Humped
            # Curve rises then falls
            for i in range(1, 7):
                if i <= 3:
                    term_structure['months'][f'M{i}'] = current_vix * (1 + 0.03 * i + np.random.uniform(-0.01, 0.01))
                else:
                    term_structure['months'][f'M{i}'] = term_structure['months'][f'M3'] * (1 - 0.03 * (i-3) + np.random.uniform(-0.01, 0.01))
                    
        return term_structure
        
    def _check_for_setups(self, vix_data, calendar, yield_data, spx_iv, spx_rv, term_structure):
        """
        Check for VIX trading opportunities
        
        Args:
            vix_data: DataFrame with VIX price data
            calendar: Economic calendar events
            yield_data: Yield curve data
            spx_iv: S&P 500 implied volatility
            spx_rv: S&P 500 realized volatility
            term_structure: VIX futures term structure
            
        Returns:
            TradeSetup with trade details or no-trade reason
        """
        # Check for IV-RV divergence setup
        iv_rv_setup = self._check_iv_rv_divergence(vix_data, spx_iv, spx_rv)
        if iv_rv_setup:
            return iv_rv_setup
            
        # Check for term structure opportunity
        term_structure_setup = self._check_term_structure(vix_data, term_structure)
        if term_structure_setup:
            return term_structure_setup
            
        # Check for event-driven volatility play
        event_setup = self._check_event_volatility(vix_data, calendar)
        if event_setup:
            return event_setup
            
        # Check for extreme volatility mean reversion
        extreme_vol_setup = self._check_extreme_volatility(vix_data)
        if extreme_vol_setup:
            return extreme_vol_setup
            
        # No valid setup found
        return TradeSetup.no_trade(
            instrument=self.instrument,
            strategy=self.name,
            reason="No actionable volatility setups detected under current market conditions"
        )
        
    def _check_iv_rv_divergence(self, vix_data, spx_iv, spx_rv):
        """
        Check for significant divergence between implied and realized volatility
        
        Returns:
            TradeSetup if valid opportunity exists, otherwise None
        """
        # Check if there's a significant divergence between IV and RV
        iv_rv_ratio = spx_iv / spx_rv if spx_rv > 0 else float('inf')
        
        # Current VIX value
        current_vix = vix_data.iloc[-1]['close']
        
        # Significant IV > RV (overpriced options)
        if iv_rv_ratio > 1.3:
            direction = "Short"  # Volatility is likely to decrease
            
            # Calculate entry, stop and targets
            entry_zone = (current_vix * 0.98, current_vix * 1.02)
            stop_loss = current_vix * 1.15  # 15% stop
            targets = [
                current_vix * 0.85,  # Target 1: 15% decrease
                spx_rv * 1.1  # Target 2: Near realized volatility
            ]
            
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Generate rationale
            rationale = (
                f"Significant IV-RV divergence detected: IV/RV ratio = {iv_rv_ratio:.2f}. "
                f"SPX implied volatility ({spx_iv:.1f}) is substantially higher than "
                f"realized volatility ({spx_rv:.1f}). This suggests volatility is overpriced "
                f"and likely to mean-revert downward. Strategy: Short VIX or sell SPX options spreads."
            )
            
            # Determine confidence based on divergence magnitude
            confidence = min(85, 50 + (iv_rv_ratio - 1) * 50)
            
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
            
        # Significant RV > IV (underpriced options)
        elif iv_rv_ratio < 0.8:
            direction = "Long"  # Volatility is likely to increase
            
            # Calculate entry, stop and targets
            entry_zone = (current_vix * 0.98, current_vix * 1.02)
            stop_loss = current_vix * 0.85  # 15% stop
            targets = [
                current_vix * 1.15,  # Target 1: 15% increase
                spx_rv * 0.9  # Target 2: Near realized volatility
            ]
            
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Generate rationale
            rationale = (
                f"Significant IV-RV divergence detected: IV/RV ratio = {iv_rv_ratio:.2f}. "
                f"SPX implied volatility ({spx_iv:.1f}) is substantially lower than "
                f"realized volatility ({spx_rv:.1f}). This suggests volatility is underpriced "
                f"and likely to mean-revert upward. Strategy: Long VIX or buy SPX options straddles/strangles."
            )
            
            # Determine confidence based on divergence magnitude
            confidence = min(85, 50 + ((1/iv_rv_ratio) - 1) * 50)
            
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
        
    def _check_term_structure(self, vix_data, term_structure):
        """
        Check for opportunities based on VIX futures term structure
        
        Returns:
            TradeSetup if valid opportunity exists, otherwise None
        """
        if not term_structure:
            return None
            
        # Current VIX value
        current_vix = vix_data.iloc[-1]['close']
        
        # Term structure shape
        shape = term_structure.get('shape', 'unknown')
        
        # Get the first few months of the curve
        try:
            m1 = term_structure['months'].get('M1', current_vix)
            m2 = term_structure['months'].get('M2', current_vix)
            m3 = term_structure['months'].get('M3', current_vix)
            
            # Calculate steepness metrics
            front_month_spread = (m1 - current_vix) / current_vix
            curve_steepness = (m3 - m1) / m1
        except:
            return None
            
        # Check for steep contango (good for short vol)
        if shape == 'contango' and curve_steepness > 0.05:
            direction = "Short"
            
            # Calculate entry, stop and targets
            entry_zone = (current_vix * 0.98, current_vix * 1.02)
            stop_loss = current_vix * 1.15  # 15% stop
            
            # Target 1: Midpoint between spot and M1
            # Target 2: Close to M1 level
            targets = [
                current_vix + (m1 - current_vix) * 0.5,
                m1 * 0.95
            ]
            
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Generate rationale
            rationale = (
                f"Steep contango detected in VIX term structure: curve steepness = {curve_steepness:.2f}. "
                f"Front-month premium = {front_month_spread:.2f}. "
                f"Term structure indicates market expects higher future volatility, but "
                f"contango provides good opportunity for short volatility positions "
                f"to benefit from time decay. Strategy: Short VIX or VIX futures."
            )
            
            # Determine confidence based on curve steepness
            confidence = min(80, 50 + curve_steepness * 400)
            
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
            
        # Check for backwardation (good for long vol)
        elif shape == 'backwardation' and front_month_spread < -0.03:
            direction = "Long"
            
            # Calculate entry, stop and targets
            entry_zone = (current_vix * 0.98, current_vix * 1.02)
            stop_loss = current_vix * 0.85  # 15% stop
            
            # Target 1: 15% increase
            # Target 2: 25% increase (significant vol spike)
            targets = [
                current_vix * 1.15,
                current_vix * 1.25
            ]
            
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Generate rationale
            rationale = (
                f"Backwardation detected in VIX term structure: front-month spread = {front_month_spread:.2f}. "
                f"Term structure indicates market stress and expectation of declining volatility, "
                f"but backwardation often precedes further volatility spikes. "
                f"Strategy: Long VIX or buy SPX options straddles/strangles."
            )
            
            # Determine confidence based on backwardation steepness
            confidence = min(80, 50 + abs(front_month_spread) * 500)
            
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
    
    def _check_event_volatility(self, vix_data, calendar):
        """
        Check for event-driven volatility opportunities
        
        Returns:
            TradeSetup if valid opportunity exists, otherwise None
        """
        if not calendar:
            return None
            
        # Current time
        now = datetime.datetime.now()
        
        # Current VIX value
        current_vix = vix_data.iloc[-1]['close']
        
        # Filter calendar for volatility-impacting events
        volatility_events = []
        for event in calendar:
            if 'event' not in event or 'time' not in event:
                continue
                
            # Check if event is volatility-impacting
            is_vol_event = any(key in event['event'] for key in self.volatility_events)
            high_impact = event.get('impact', 0) >= 2  # On a scale of 0-3
            
            if is_vol_event and high_impact:
                # Parse event time
                try:
                    event_time = datetime.datetime.fromtimestamp(event['time'])
                    days_until_event = (event_time - now).total_seconds() / (24 * 3600)
                    
                    # Only consider events in the next 10 days
                    if 0 < days_until_event < 10:
                        event['days_until'] = days_until_event
                        volatility_events.append(event)
                except:
                    continue
                    
        # If high-impact events coming up
        if volatility_events:
            # Sort by proximity
            volatility_events.sort(key=lambda x: x.get('days_until', 10))
            next_event = volatility_events[0]
            days_until = next_event.get('days_until', 0)
            
            # Check current VIX level relative to historical and event proximity
            # If VIX is low before a major event, go long volatility
            if current_vix < self.low_vol_threshold and days_until < 5:
                direction = "Long"
                
                # Calculate entry, stop and targets
                entry_zone = (current_vix * 0.98, current_vix * 1.02)
                stop_loss = current_vix * 0.9  # 10% stop
                
                # Target 1: 15% increase
                # Target 2: 25% increase (event volatility)
                targets = [
                    current_vix * 1.15,
                    current_vix * 1.25
                ]
                
                # Calculate risk/reward
                risk_reward = TradeSetup.calculate_risk_reward(
                    entry=sum(entry_zone) / 2,
                    stop=stop_loss,
                    target=targets[0]
                )
                
                # Generate rationale
                event_time_str = datetime.datetime.fromtimestamp(next_event['time']).strftime('%Y-%m-%d')
                rationale = (
                    f"Low VIX ({current_vix:.1f}) with high-impact event {next_event['event']} "
                    f"approaching on {event_time_str} ({days_until:.1f} days away). "
                    f"Current implied volatility levels appear underpriced relative to typical "
                    f"pre-event volatility. Strategy: Long volatility through VIX calls or "
                    f"SPX options straddles/strangles."
                )
                
                # Determine confidence based on event proximity and impact
                event_impact = next_event.get('impact', 0)
                proximity_factor = max(0, 5 - days_until) / 5  # 0 to 1 scale
                confidence = min(85, 55 + (event_impact * 10) + (proximity_factor * 20))
                
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
        
    def _check_extreme_volatility(self, vix_data):
        """
        Check for extreme volatility mean reversion opportunities
        
        Returns:
            TradeSetup if valid opportunity exists, otherwise None
        """
        if vix_data.shape[0] < 10:
            return None
            
        # Current VIX value
        current_vix = vix_data.iloc[-1]['close']
        
        # Calculate recent VIX metrics
        vix_5d_avg = vix_data['close'].iloc[-5:].mean()
        vix_20d_avg = vix_data['close'].iloc[-20:].mean() if vix_data.shape[0] >= 20 else vix_5d_avg
        
        # Check for extreme high volatility
        if current_vix > self.extreme_vol_threshold and current_vix > vix_20d_avg * 1.4:
            direction = "Short"  # Volatility is extremely high, likely to mean-revert down
            
            # Calculate entry, stop and targets
            entry_zone = (current_vix * 0.98, current_vix * 1.02)
            stop_loss = current_vix * 1.15  # 15% stop
            
            # Target 1: 20% decrease
            # Target 2: 20-day average
            targets = [
                current_vix * 0.8,
                vix_20d_avg
            ]
            
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Generate rationale
            rationale = (
                f"Extreme volatility detected: VIX at {current_vix:.1f}, {(current_vix/vix_20d_avg - 1)*100:.1f}% "
                f"above 20-day average of {vix_20d_avg:.1f}. "
                f"Volatility typically mean-reverts after extreme spikes. "
                f"Strategy: Short VIX or implement negative vega option spreads."
            )
            
            # Determine confidence based on deviation from average
            deviation = current_vix / vix_20d_avg - 1
            confidence = min(85, 50 + deviation * 100)
            
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
            
        # Check for extreme low volatility with potential for increase
        elif current_vix < self.low_vol_threshold and current_vix < vix_20d_avg * 0.8:
            direction = "Long"  # Volatility is extremely low, potential for spike
            
            # Calculate entry, stop and targets
            entry_zone = (current_vix * 0.98, current_vix * 1.02)
            stop_loss = current_vix * 0.9  # 10% stop
            
            # Target 1: 20% increase
            # Target 2: 20-day average
            targets = [
                current_vix * 1.2,
                vix_20d_avg
            ]
            
            # Calculate risk/reward
            risk_reward = TradeSetup.calculate_risk_reward(
                entry=sum(entry_zone) / 2,
                stop=stop_loss,
                target=targets[0]
            )
            
            # Generate rationale
            rationale = (
                f"Extremely low volatility detected: VIX at {current_vix:.1f}, {(1-current_vix/vix_20d_avg)*100:.1f}% "
                f"below 20-day average of {vix_20d_avg:.1f}. "
                f"Low volatility environments often precede volatility expansion. "
                f"Strategy: Long VIX or implement positive gamma option strategies."
            )
            
            # Determine confidence based on deviation from average
            deviation = 1 - current_vix / vix_20d_avg
            confidence = min(80, 50 + deviation * 100)
            
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