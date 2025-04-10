import datetime
import threading
import time
import logging
import json
from pathlib import Path
import websocket

from src.strategies.gbpusd_scalping import GBPUSDScalpingStrategy
from src.strategies.usdjpy_event import USDJPYEventStrategy
from src.strategies.vix_volatility import VIXVolatilityStrategy
from src.strategies.crypto_strategy import CryptoStrategy
from src.utils.logger import Logger

class TradingAgent:
    """
    Autonomous trading agent that analyzes markets using multiple strategies
    and generates trade setups with real-time capabilities
    """
    
    def __init__(self, data_fetcher, real_time_mode=False, update_interval=60):
        """
        Initialize the trading agent
        
        Args:
            data_fetcher: DataFetcher instance for retrieving market data
            real_time_mode: Whether to run in real-time mode
            update_interval: Interval for real-time updates in seconds
        """
        self.data_fetcher = data_fetcher
        self.logger = Logger()
        
        # Real-time mode settings
        self.real_time_mode = real_time_mode
        self.update_interval = update_interval
        self.real_time_running = False
        self.real_time_thread = None
        self.price_alerts = {}
        self.watched_symbols = set()
        
        # Create results directory
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize all strategies
        self.strategies = [
            GBPUSDScalpingStrategy(data_fetcher),
            USDJPYEventStrategy(data_fetcher),
            VIXVolatilityStrategy(data_fetcher),
            CryptoStrategy(data_fetcher)
        ]
        
        self.last_analysis_time = None
        self.recent_trade_setups = {}
        
        # If real-time mode is enabled, start real-time monitoring
        if real_time_mode:
            self.start_real_time_monitoring()
    
    def start_real_time_monitoring(self):
        """Start real-time price monitoring for supported symbols"""
        if self.real_time_running:
            self.logger.warning("Real-time monitoring already running")
            return
            
        self.real_time_running = True
        
        # Get all symbols from strategies
        symbols = []
        for strategy in self.strategies:
            if hasattr(strategy, 'instrument'):
                symbols.append(strategy.instrument)
            elif hasattr(strategy, 'instruments'):
                symbols.extend(strategy.instruments)
        
        # Remove duplicates
        self.watched_symbols = set(symbols)
        self.logger.info(f"Watching symbols: {', '.join(self.watched_symbols)}")
        
        # Subscribe to real-time updates with error handling
        successful_subscriptions = 0
        for symbol in self.watched_symbols:
            try:
                if self.data_fetcher.subscribe_to_real_time(symbol, self._on_price_update):
                    successful_subscriptions += 1
                else:
                    self.logger.warning(f"Failed to subscribe to {symbol}")
            except Exception as e:
                self.logger.error(f"Error subscribing to {symbol}: {e}")
        
        self.logger.info(f"Successfully subscribed to {successful_subscriptions} out of {len(self.watched_symbols)} symbols")
        
        # Start real-time thread
        self.real_time_thread = threading.Thread(target=self._real_time_loop, daemon=True)
        self.real_time_thread.start()
        
        self.logger.info("Real-time monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time price monitoring"""
        if not self.real_time_running:
            return
            
        self.real_time_running = False
        
        # Unsubscribe from all symbols
        for symbol in self.watched_symbols:
            self.data_fetcher.unsubscribe_from_real_time(symbol)
            
        # Wait for thread to stop
        if self.real_time_thread:
            self.real_time_thread.join(timeout=5)
            
        self.logger.info("Real-time monitoring stopped")
    
    def _real_time_loop(self):
        """Background thread for real-time analysis"""
        while self.real_time_running:
            try:
                # Run quick analysis
                self._run_real_time_analysis()
                
                # Sleep until next update
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in real-time loop: {e}")
    
    def _on_price_update(self, symbol, price, volume, timestamp):
        """Callback for real-time price updates"""
        # Check if we have any price alerts for this symbol
        if symbol in self.price_alerts:
            for alert_id, alert in list(self.price_alerts[symbol].items()):
                # Check if alert condition is met
                if alert['direction'] == 'above' and price >= alert['price']:
                    self.logger.info(f"ALERT: {symbol} crossed above {alert['price']}")
                    if alert['callback']:
                        alert['callback'](symbol, price, 'above')
                    # Remove alert once triggered
                    del self.price_alerts[symbol][alert_id]
                elif alert['direction'] == 'below' and price <= alert['price']:
                    self.logger.info(f"ALERT: {symbol} crossed below {alert['price']}")
                    if alert['callback']:
                        alert['callback'](symbol, price, 'below')
                    # Remove alert once triggered
                    del self.price_alerts[symbol][alert_id]
    
    def set_price_alert(self, symbol, price, direction, callback=None):
        """
        Set a price alert for a symbol
        
        Args:
            symbol: Symbol to watch
            price: Price level to alert at
            direction: 'above' or 'below'
            callback: Function to call when alert triggers
            
        Returns:
            Alert ID
        """
        if symbol not in self.price_alerts:
            self.price_alerts[symbol] = {}
            
        # Generate unique ID
        alert_id = f"{symbol}_{direction}_{price}_{int(time.time())}"
        
        # Store alert
        self.price_alerts[symbol][alert_id] = {
            'price': price,
            'direction': direction,
            'callback': callback,
            'created_at': datetime.datetime.now()
        }
        
        self.logger.info(f"Price alert set: {symbol} {direction} {price}")
        return alert_id
    
    def cancel_price_alert(self, alert_id):
        """Cancel a price alert by ID"""
        for symbol in self.price_alerts:
            if alert_id in self.price_alerts[symbol]:
                del self.price_alerts[symbol][alert_id]
                self.logger.info(f"Price alert cancelled: {alert_id}")
                return True
        
        return False
    
    def _run_real_time_analysis(self):
        """Run a quick real-time analysis of current market conditions"""
        try:
            # Get latest prices for all watched symbols
            current_prices = {}
            for symbol in self.watched_symbols:
                price_data = self.data_fetcher.get_real_time_price(symbol)
                if price_data:
                    current_prices[symbol] = price_data
            
            # Check recent trade setups for entry/exit opportunities
            for setup_name, setup in self.recent_trade_setups.items():
                if not setup or setup.direction == "No Trade":
                    continue
                    
                symbol = setup.instrument
                if symbol not in current_prices:
                    continue
                    
                current_price = current_prices[symbol]['price']
                
                # Check if we're in the entry zone
                if setup.entry_zone and setup.entry_zone[0] <= current_price <= setup.entry_zone[1]:
                    self.logger.info(f"REAL-TIME ALERT: {symbol} in entry zone {setup.entry_zone[0]:.5f}-{setup.entry_zone[1]:.5f}")
                
                # Check if we've hit stop loss
                if setup.stop_loss:
                    if (setup.direction == "Long" and current_price <= setup.stop_loss) or \
                       (setup.direction == "Short" and current_price >= setup.stop_loss):
                        self.logger.warning(f"REAL-TIME ALERT: {symbol} hit stop loss at {setup.stop_loss:.5f}")
                
                # Check if we've hit targets
                if setup.targets:
                    for i, target in enumerate(setup.targets):
                        if (setup.direction == "Long" and current_price >= target) or \
                           (setup.direction == "Short" and current_price <= target):
                            self.logger.info(f"REAL-TIME ALERT: {symbol} reached target {i+1} at {target:.5f}")
            
        except Exception as e:
            self.logger.error(f"Error in real-time analysis: {e}")
    
    def analyze_markets(self):
        """
        Run all strategies to analyze the markets and generate trade setups
        
        Returns:
            dict: Dictionary mapping strategy names to trade setups
        """
        self.logger.info("Starting market analysis")
        self.last_analysis_time = datetime.datetime.now()
        
        # Store all trade setups
        trade_setups = {}
        
        # Run each strategy
        for strategy in self.strategies:
            try:
                self.logger.info(f"Running {strategy.name} strategy...")
                
                # Run the strategy analysis
                result = strategy.analyze()
                
                # Handle results based on type
                if result is None:
                    self.logger.warning(f"No results from {strategy.name} strategy")
                    continue
                    
                # Handle strategies that return multiple setups vs. single setup
                if isinstance(result, list):
                    # Multiple trade setups
                    for i, setup in enumerate(result):
                        if setup:
                            key = f"{setup.instrument}_{strategy.name}_{i}"
                            trade_setups[key] = setup
                            # Log trade setup
                            self.logger.log_trade_setup(setup)
                else:
                    # Single trade setup
                    setup = result
                    if setup:
                        trade_setups[f"{setup.instrument}_{strategy.name}"] = setup
                        # Log trade setup
                        self.logger.log_trade_setup(setup)
                
            except Exception as e:
                self.logger.error(f"Error running {strategy.name} strategy: {e}")
                # If a strategy fails, continue with others
                continue
                
        # Generate market commentary
        market_commentary = self._generate_market_commentary(trade_setups)
        self.logger.log_market_commentary(market_commentary)
        
        # Store recent setups
        self.recent_trade_setups = trade_setups
        
        # Setup real-time price alerts for active trade setups
        if self.real_time_mode:
            self._setup_alerts_for_trade_setups(trade_setups)
        
        # Save analysis results
        self._save_analysis_results(trade_setups, market_commentary)
        
        # Print results to console
        self._print_analysis_results(trade_setups, market_commentary)
        
        return trade_setups
    
    def _setup_alerts_for_trade_setups(self, trade_setups):
        """Setup price alerts for active trade setups"""
        # Clear old alerts
        self.price_alerts = {}
        
        for setup_name, setup in trade_setups.items():
            if not setup or setup.direction == "No Trade":
                continue
                
            symbol = setup.instrument
            
            # Set entry zone alerts
            if setup.entry_zone:
                # Alert when price enters the entry zone
                self.set_price_alert(
                    symbol, 
                    setup.entry_zone[0], 
                    'below' if setup.direction == 'Long' else 'above'
                )
                self.set_price_alert(
                    symbol, 
                    setup.entry_zone[1], 
                    'above' if setup.direction == 'Long' else 'below'
                )
            
            # Set stop loss alert
            if setup.stop_loss:
                self.set_price_alert(
                    symbol,
                    setup.stop_loss,
                    'below' if setup.direction == 'Long' else 'above'
                )
            
            # Set target alerts
            if setup.targets:
                for target in setup.targets:
                    self.set_price_alert(
                        symbol,
                        target,
                        'above' if setup.direction == 'Long' else 'below'
                    )
    
    def _save_analysis_results(self, trade_setups, market_commentary):
        """Save analysis results to disk for record-keeping"""
        try:
            # Create timestamp string
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save results to JSON
            results = {
                "timestamp": timestamp,
                "analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                "market_commentary": market_commentary,
                "trade_setups": {
                    name: setup.to_dict() if setup else None
                    for name, setup in trade_setups.items()
                }
            }
            
            # Save to file
            results_file = self.results_dir / f"analysis_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.logger.info(f"Analysis results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {e}")
    
    def _generate_market_commentary(self, trade_setups):
        """
        Generate market commentary based on trade setups
        
        Args:
            trade_setups: Dictionary mapping strategy names to trade setups
            
        Returns:
            str: Market commentary
        """
        commentary = "----- MARKET COMMENTARY -----\n"
        
        # Count valid trade setups
        valid_setups = sum(1 for setup in trade_setups.values() 
                          if setup and setup.direction != "No Trade")
        
        # Generate overall market sentiment
        if valid_setups >= 2:
            if all(setup.direction == "Long" for setup in trade_setups.values() 
                  if setup and setup.direction != "No Trade"):
                sentiment = "bullish"
            elif all(setup.direction == "Short" for setup in trade_setups.values() 
                    if setup and setup.direction != "No Trade"):
                sentiment = "bearish"
            else:
                sentiment = "mixed"
        else:
            sentiment = "neutral"
            
        commentary += f"Overall market sentiment: {sentiment.upper()}\n\n"
        
        # Add time and date
        now = datetime.datetime.now()
        commentary += f"Analysis time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Group setups by asset class
        forex_setups = []
        crypto_setups = []
        vix_setups = []
        
        for setup_name, setup in trade_setups.items():
            if setup:
                if setup.instrument in ["GBPUSD", "USDJPY"]:
                    forex_setups.append(setup)
                elif setup.instrument in ["SOLUSD", "ETHUSD"]:
                    crypto_setups.append(setup)
                elif setup.instrument == "VIX":
                    vix_setups.append(setup)
        
        # Add forex market commentary
        if forex_setups:
            commentary += "FOREX MARKET:\n"
            for setup in forex_setups:
                if setup.direction != "No Trade":
                    commentary += f"{setup.instrument}: {setup.direction} opportunity with {setup.confidence}% confidence. "
                    commentary += f"Risk/reward ratio: {setup.risk_reward:.2f if setup.risk_reward else 'N/A'}. "
                    commentary += f"{setup.rationale}\n\n"
                else:
                    commentary += f"{setup.instrument}: No trade setup. {setup.rationale}\n\n"
        
        # Add crypto market commentary
        if crypto_setups:
            commentary += "CRYPTOCURRENCY MARKET:\n"
            for setup in crypto_setups:
                if setup.direction != "No Trade":
                    commentary += f"{setup.instrument}: {setup.direction} opportunity with {setup.confidence}% confidence. "
                    commentary += f"Risk/reward ratio: {setup.risk_reward:.2f if setup.risk_reward else 'N/A'}. "
                    commentary += f"{setup.rationale}\n\n"
                else:
                    commentary += f"{setup.instrument}: No trade setup. {setup.rationale}\n\n"
        
        # Add VIX market commentary
        if vix_setups and vix_setups[0]:
            setup = vix_setups[0]
            commentary += "VOLATILITY MARKET:\n"
            if setup.direction != "No Trade":
                commentary += f"{setup.instrument}: {setup.direction} opportunity with {setup.confidence}% confidence. "
                commentary += f"Risk/reward ratio: {setup.risk_reward:.2f if setup.risk_reward else 'N/A'}. "
                commentary += f"{setup.rationale}\n\n"
            else:
                commentary += f"{setup.instrument}: No trade setup. {setup.rationale}\n\n"
                    
        # Add general market context if available
        try:
            # Get VIX data as general market risk indicator
            vix_data = self.data_fetcher.get_vix_data(days=5)
            if not vix_data.empty:
                latest_vix = vix_data.iloc[-1]['close']
                vix_change = (latest_vix / vix_data.iloc[-2]['close'] - 1) * 100  # Daily % change
                
                commentary += f"Market Volatility (VIX): {latest_vix:.2f} ({vix_change:+.2f}%). "
                
                if latest_vix < 15:
                    commentary += "Low volatility environment - favorable for trend following strategies.\n"
                elif latest_vix > 30:
                    commentary += "High volatility environment - increased risk, favor mean-reversion strategies.\n"
                else:
                    commentary += "Moderate volatility environment.\n"
        except:
            pass
            
        return commentary
        
    def _print_analysis_results(self, trade_setups, market_commentary):
        """
        Print analysis results to console
        
        Args:
            trade_setups: Dictionary mapping strategy names to trade setups
            market_commentary: Market commentary string
        """
        print("\n" + "="*80)
        print(" "*30 + "TRADING AGENT ANALYSIS")
        print("="*80 + "\n")
        
        # Print each trade setup
        for strategy_name, setup in trade_setups.items():
            if setup:
                print("-"*80)
                print(str(setup))
                print("-"*80 + "\n")
                
        # Print market commentary
        print(market_commentary)
        print("="*80 + "\n")
    
    def to_json(self):
        """
        Convert trading agent state to JSON
        
        Returns:
            dict: Trading agent state in JSON format
        """
        return {
            "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "trade_setups": {
                name: setup.to_dict() if setup else None
                for name, setup in self.recent_trade_setups.items()
            },
            "real_time_mode": self.real_time_mode,
            "real_time_running": self.real_time_running,
            "watched_symbols": list(self.watched_symbols),
            "price_alerts_count": sum(len(alerts) for alerts in self.price_alerts.values()) if self.price_alerts else 0
        }

    def subscribe_to_real_time(self, symbol, callback):
        """
        Subscribe to real-time updates for a symbol
        
        Args:
            symbol: Trading symbol to subscribe to
            callback: Function to call on price updates
        """
        if self.ws is None:
            self.logger.warning("Websocket connection not available")
            return False
        
        try:
            # Map symbol to provider format
            provider_symbol = self._map_to_provider_symbol(symbol)
            
            # Add to subscribers
            if provider_symbol not in self.real_time_subscribers:
                self.real_time_subscribers[provider_symbol] = []
                # Subscribe to symbol
                self.ws.send(json.dumps({'type': 'subscribe', 'symbol': provider_symbol}))
            
            self.real_time_subscribers[provider_symbol].append(callback)
            return True
        except websocket._exceptions.WebSocketConnectionClosedException:
            self.logger.warning(f"Cannot subscribe to {symbol} - WebSocket connection closed")
            # Attempt to reconnect
            self._setup_real_time_connection()
            return False
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {e}")
            return False 