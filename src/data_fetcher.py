import os
import time
import datetime
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import finnhub
import json
import hashlib
from pathlib import Path
import threading
import logging

# Import alternative data providers
try:
    from alternative_data_providers import AlternativeDataProvider
    ALTERNATIVE_PROVIDERS_AVAILABLE = True
except ImportError:
    ALTERNATIVE_PROVIDERS_AVAILABLE = False

class DataFetcher:
    """Enhanced class for fetching market data from various sources with production-ready features"""
    
    def __init__(self, api_key="demo", use_cache=True, cache_expiry=3600, api_secret=None):
        """
        Initialize data fetcher with enhanced features
        
        Args:
            api_key: API key for Finnhub or other data providers
            use_cache: Whether to use local cache for API requests
            cache_expiry: Cache expiry time in seconds (default 1 hour)
            api_secret: Secret for Finnhub authentication (optional)
        """
        self.finnhub_client = finnhub.Client(api_key=api_key)
        self.api_key = api_key
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'X-Finnhub-Token': api_key
        }
        
        # Add Finnhub secret if provided
        if api_secret:
            self.headers['X-Finnhub-Secret'] = api_secret
        
        # Setup caching
        self.use_cache = use_cache
        self.cache_expiry = cache_expiry
        self.cache_dir = Path('data/cache')
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Setup rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self.request_lock = threading.Lock()
        
        # Setup logging with proper configuration
        self.logger = logging.getLogger('DataFetcher')
        # Get log level from environment
        log_level = os.getenv("LOG_LEVEL", "INFO")
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure logger if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(level)
            
            # Also add file handler
            file_handler = logging.FileHandler('logs/data_fetcher.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"DataFetcher initialized with API key: {'demo' if api_key == 'demo' else 'valid key'}")
        self.logger.info(f"Cache enabled: {use_cache}, Expiry: {cache_expiry} seconds")
        self.logger.info(f"Log level set to: {log_level}")
        
        # Initialize alternative data providers if available
        self.alt_provider = None
        if ALTERNATIVE_PROVIDERS_AVAILABLE:
            try:
                self.alt_provider = AlternativeDataProvider(use_cache=use_cache, cache_expiry=cache_expiry)
                self.logger.info("Alternative data providers initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize alternative data providers: {e}")
        
        # Last fetched real-time prices
        self.last_prices = {}
        
        # Initialize websocket connection for real-time data if not using demo
        self.ws = None
        self.real_time_subscribers = {}
        if api_key != "demo":
            self._setup_real_time_connection()
    
    def _setup_real_time_connection(self):
        """Setup websocket connection for real-time data"""
        try:
            import websocket
            
            # Setup websocket connection
            socket_url = f"wss://ws.finnhub.io?token={self.api_key}"
            
            def on_message(ws, message):
                """Handle incoming WebSocket messages."""
                try:
                    # Parse the message
                    msg = json.loads(message)
                    
                    # Send acknowledgment to prevent timeouts
                    if ws and ws.sock and ws.sock.connected:
                        ws.send(json.dumps({"type": "ping"}))
                        self.logger.debug("Sent ping acknowledgment")
                    
                    # Process the message
                    if "type" in msg:
                        # Process trade data
                        if msg["type"] == "trade":
                            for trade in msg["data"]:
                                symbol = trade["s"]
                                price = trade["p"]
                                volume = trade["v"]
                                timestamp = trade["t"]
                                
                                # Update last price
                                self.last_prices[symbol] = {
                                    'price': price,
                                    'volume': volume,
                                    'timestamp': timestamp
                                }
                                
                                # Notify subscribers
                                if symbol in self.real_time_subscribers:
                                    for callback in self.real_time_subscribers[symbol]:
                                        try:
                                            callback(symbol, price, volume, timestamp)
                                        except Exception as e:
                                            self.logger.error(f"Error in callback for {symbol}: {e}")
                except json.JSONDecodeError:
                    self.logger.warning(f"Received invalid JSON: {message}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
            
            def on_error(ws, error):
                self.logger.error(f"Websocket error: {error}")
                
            def on_close(ws, close_status_code, close_msg):
                self.logger.info(f"Websocket connection closed: {close_status_code} {close_msg}")
                # Attempt to reconnect after delay
                self.ws = None
                self.logger.info("Will attempt to reconnect after 15 seconds")
                threading.Timer(15, self._setup_real_time_connection).start()
                
            def on_open(ws):
                self.logger.info("Websocket connection established")
                # Subscribe to default symbols
                for symbol in ['OANDA:EUR_USD', 'OANDA:GBP_USD', 'BINANCE:ETHUSDT', 'BINANCE:SOLUSDT']:
                    try:
                        ws.send(json.dumps({'type': 'subscribe', 'symbol': symbol}))
                        self.logger.info(f"Subscribed to {symbol}")
                    except Exception as e:
                        self.logger.error(f"Error subscribing to {symbol}: {e}")
            
            # Create websocket connection with ping interval to keep connection alive
            self.ws = websocket.WebSocketApp(
                socket_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Set ping interval to keep connection alive (15 seconds)
            websocket_thread = threading.Thread(
                target=lambda: self.ws.run_forever(
                    ping_interval=15,
                    ping_timeout=10,
                    ping_payload='{"type":"ping"}'
                ),
                daemon=True
            )
            websocket_thread.start()
            self.logger.info(f"WebSocket thread started with ping interval of 15 seconds")
            
        except ImportError:
            self.logger.warning("websocket-client not installed. Real-time data unavailable.")
        except Exception as e:
            self.logger.error(f"Error setting up real-time connection: {e}")
    
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
                try:
                    if self.ws.sock and self.ws.sock.connected:
                        self.ws.send(json.dumps({'type': 'subscribe', 'symbol': provider_symbol}))
                        self.logger.info(f"Subscribed to {provider_symbol}")
                    else:
                        self.logger.warning(f"Socket not connected, cannot subscribe to {provider_symbol}")
                        # Try to reconnect
                        self._setup_real_time_connection()
                        return False
                except Exception as e:
                    self.logger.error(f"Error sending subscription request for {provider_symbol}: {e}")
                    return False
                
            self.real_time_subscribers[provider_symbol].append(callback)
            return True
        except Exception as e:
            self.logger.error(f"Unexpected error in subscribe_to_real_time for {symbol}: {e}")
            return False
    
    def unsubscribe_from_real_time(self, symbol, callback=None):
        """
        Unsubscribe from real-time updates
        
        Args:
            symbol: Trading symbol to unsubscribe from
            callback: Specific callback to remove, or None to remove all
        """
        provider_symbol = self._map_to_provider_symbol(symbol)
        
        if provider_symbol in self.real_time_subscribers:
            if callback:
                # Remove specific callback
                self.real_time_subscribers[provider_symbol].remove(callback)
            else:
                # Remove all callbacks
                self.real_time_subscribers[provider_symbol] = []
                
            # If no more callbacks, unsubscribe
            if not self.real_time_subscribers[provider_symbol] and self.ws:
                self.ws.send(json.dumps({'type': 'unsubscribe', 'symbol': provider_symbol}))
    
    def _map_to_provider_symbol(self, symbol):
        """Map our symbol format to the data provider's format"""
        symbol_map = {
            'EURUSD': 'OANDA:EUR_USD',
            'GBPUSD': 'OANDA:GBP_USD',
            'USDJPY': 'OANDA:USD_JPY',
            'ETHUSD': 'BINANCE:ETHUSDT',
            'SOLUSD': 'BINANCE:SOLUSDT',
            'BTCUSD': 'BINANCE:BTCUSDT',
            'ADAUSD': 'BINANCE:ADAUSDT',
            'DOTUSD': 'BINANCE:DOTUSDT',
            'LINKUSD': 'BINANCE:LINKUSDT'
        }
        return symbol_map.get(symbol, symbol)
    
    def _rate_limit_request(self):
        """Implement rate limiting for API requests"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
                
            self.last_request_time = time.time()
    
    def _get_cache_key(self, params):
        """Generate a cache key from request parameters"""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key):
        """Get data from cache if available and not expired"""
        if not self.use_cache:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                # Check if cache is expired
                if time.time() - cached_data['timestamp'] < self.cache_expiry:
                    return cached_data['data']
            except Exception as e:
                self.logger.warning(f"Error reading from cache: {e}")
                
        return None
    
    def _save_to_cache(self, cache_key, data):
        """Save data to cache"""
        if not self.use_cache:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cached_data = {
                'timestamp': time.time(),
                'data': data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f)
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {e}")
    
    def get_forex_data(self, symbol, resolution='15', count=100, use_cache=True):
        """
        Fetch forex or crypto data for a given symbol with caching and error handling
        
        Args:
            symbol: Forex/crypto symbol (e.g., 'GBPUSD', 'ETHUSD')
            resolution: Time resolution (1, 5, 15, 30, 60, D, W, M)
            count: Number of candles to fetch
            use_cache: Whether to use cache for this specific request
            
        Returns:
            Pandas DataFrame with OHLCV data
        """
        # Calculate time periods
        to_time = int(time.time())
        from_time = to_time - (int(resolution) * 60 * count) if resolution.isdigit() else to_time - (86400 * count)
        
        # Create cache parameters
        cache_params = {
            'type': 'forex_data',
            'symbol': symbol,
            'resolution': resolution,
            'from_time': from_time,
            'to_time': to_time
        }
        
        cache_key = self._get_cache_key(cache_params)
        
        # Check cache if enabled for this request
        if use_cache and self.use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                try:
                    df = pd.DataFrame(cached_data)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    return df
                except Exception as e:
                    self.logger.warning(f"Error parsing cached data: {e}")
        
        # Apply rate limiting
        self._rate_limit_request()
        
        try:
            # Handle cryptocurrency pairs
            if symbol in ['ETHUSD', 'SOLUSD']:
                data = self._get_crypto_data(symbol, resolution, from_time, to_time)
            else:
                # Fetch forex data
                data = self.finnhub_client.forex_candles(
                    symbol=symbol, 
                    resolution=resolution, 
                    _from=from_time, 
                    to=to_time
                )
            
            if data and data['s'] == 'ok':
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                
                # Save to cache if enabled
                if use_cache and self.use_cache:
                    self._save_to_cache(cache_key, df.reset_index().to_dict('records'))
                
                df.set_index('timestamp', inplace=True)
                return df
            else:
                self.logger.error(f"Error fetching data for {symbol}: {data}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_real_time_price(self, symbol):
        """
        Get real-time price for a symbol with robust fallbacks
        
        Args:
            symbol: Trading symbol
            
        Returns:
            dict with price, volume, and timestamp, or None if not available
        """
        provider_symbol = self._map_to_provider_symbol(symbol)
        
        # First check cache for very recent data (less than 1 second)
        if provider_symbol in self.last_prices:
            price_data = self.last_prices[provider_symbol]
            if time.time() - price_data['timestamp']/1000 < 1:
                return price_data
        
        # Next, try the appropriate API based on the symbol type
        result = None
        try:
            self._rate_limit_request()
            
            # Handle different symbol types
            if symbol in ['ETHUSD', 'SOLUSD']:
                # Crypto data
                result = self._get_crypto_real_time_price(symbol)
            elif symbol == 'VIX':
                # VIX data
                result = self._get_index_real_time_price(symbol)
            else:
                # Forex data
                result = self._get_forex_real_time_price(symbol)
                
            if result:
                # Cache the result and return
                self.last_prices[provider_symbol] = result
                self.logger.debug(f"Got real-time price for {symbol}: {result['price']} (source: {result.get('source', 'unknown')})")
                return result
                
        except Exception as e:
            self.logger.warning(f"Error fetching real-time price for {symbol}: {str(e)}")
            
        # If we get here, all API methods failed
        # Fall back to mock data and clearly log this
        mock_result = self._get_mock_price(symbol)
        if mock_result:
            self.logger.info(f"Using MOCK price data for {symbol}: {mock_result['price']}")
        
        return mock_result
    
    def _get_crypto_real_time_price(self, symbol):
        """Get real-time cryptocurrency price"""
        # Always return mock data for demo key
        if self.api_key == "demo":
            self.logger.debug(f"Using mock price data for {symbol} (demo mode)")
            return self._get_mock_price(symbol)
        
        # First try using the alternative provider if available
        if self.alt_provider:
            try:
                alt_price = self.alt_provider.get_real_time_crypto(symbol)
                if alt_price:
                    self.logger.debug(f"Successfully fetched {symbol} price from alternative provider: {alt_price['price']}")
                    return {
                        'price': alt_price['price'],
                        'volume': alt_price.get('volume_24h', 0),
                        'timestamp': int(time.time() * 1000),
                        'source': alt_price.get('source', 'alternative')
                    }
            except Exception as e:
                self.logger.warning(f"Error fetching crypto price from alternative provider for {symbol}: {str(e)}")
        
        # If alternative provider failed or not available, try Finnhub
        try:
            # Map to the correct Binance symbol
            binance_symbol_map = {
                'ETHUSD': 'ETHUSDT',
                'SOLUSD': 'SOLUSDT',
                'BTCUSD': 'BTCUSDT', 
                'ADAUSD': 'ADAUSDT',
                'DOTUSD': 'DOTUSDT',
                'LINKUSD': 'LINKUSDT'
            }
            binance_symbol = binance_symbol_map.get(symbol, 'ETHUSDT')
            finnhub_symbol = f'BINANCE:{binance_symbol}'
            
            # Get latest candle using crypto_candles endpoint
            to_time = int(time.time())
            from_time = to_time - 300  # Last 5 minutes
            
            candles = self.finnhub_client.crypto_candles(
                symbol=finnhub_symbol,
                resolution='1',
                _from=from_time,
                to=to_time
            )
            
            if candles and candles['s'] == 'ok' and len(candles['c']) > 0:
                self.logger.debug(f"Successfully fetched {symbol} price from Finnhub: {candles['c'][-1]}")
                return {
                    'price': candles['c'][-1],  # Latest close price
                    'volume': candles['v'][-1],  # Latest volume
                    'timestamp': int(time.time() * 1000),  # Current time
                    'source': 'finnhub'
                }
            else:
                if candles:
                    self.logger.warning(f"Invalid candles data for {symbol}: {candles}")
                else:
                    self.logger.warning(f"Empty candles response for {symbol}")
        except AttributeError as e:
            if 'crypto_quote' in str(e):
                self.logger.warning(f"The 'crypto_quote' method is not available for {symbol}. Using crypto_candles instead.")
                # Already trying crypto_candles above, fall back to mock data
                self.logger.info(f"Falling back to mock data for {symbol} due to API method error")
                return self._get_mock_price(symbol)
            else:
                self.logger.warning(f"Error fetching crypto price for {symbol}: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error fetching crypto price via candles for {symbol}: {str(e)}")
        
        # Fall back to mock data if we get here
        self.logger.info(f"Falling back to mock data for {symbol}")
        return self._get_mock_price(symbol)
    
    def _get_forex_real_time_price(self, symbol):
        """Get real-time forex price"""
        # Always return mock data for demo key
        if self.api_key == "demo":
            self.logger.debug(f"Using mock price data for {symbol} (demo mode)")
            return self._get_mock_price(symbol)
        
        try:
            # Try forex_rates endpoint
            base = symbol[:3]
            target = symbol[3:]
            
            rates = self.finnhub_client.forex_rates(base=base)
            
            if rates and target in rates:
                self.logger.debug(f"Successfully fetched {symbol} price: {rates[target]}")
                return {
                    'price': rates[target],
                    'volume': 0,  # Forex rates don't include volume
                    'timestamp': int(time.time() * 1000),
                    'source': 'api'
                }
            else:
                if rates:
                    self.logger.warning(f"Target currency {target} not found in rates response: {rates}")
                else:
                    self.logger.warning(f"Empty rates response for {symbol}")
        except Exception as e:
            self.logger.warning(f"Error fetching forex price for {symbol}: {str(e)}")
            # Check if it's an invalid API key error
            if "Invalid API key" in str(e):
                self.logger.error(f"Invalid API key detected. Forex data may not be included in your subscription plan.")
                # Fall back to mock data immediately
                return self._get_mock_price(symbol)
        
        # Fall back to mock data if we get here
        self.logger.info(f"Falling back to mock data for {symbol}")
        return self._get_mock_price(symbol)
    
    def _get_index_real_time_price(self, symbol):
        """Get real-time index price (e.g., VIX)"""
        # Always return mock data for demo key
        if self.api_key == "demo":
            self.logger.debug(f"Using mock price data for {symbol} (demo mode)")
            return self._get_mock_price(symbol)
        
        try:
            # For VIX, we can use the stock quote endpoint
            if symbol == 'VIX':
                quote = self.finnhub_client.quote('CBOE:VIX')
                
                if quote and 'c' in quote:
                    self.logger.debug(f"Successfully fetched {symbol} price: {quote['c']}")
                    return {
                        'price': quote['c'],  # Current price
                        'volume': quote.get('v', 0),  # Volume if available
                        'timestamp': int(time.time() * 1000),
                        'source': 'api'
                    }
                else:
                    if quote:
                        self.logger.warning(f"Invalid quote data for {symbol}: {quote}")
                    else:
                        self.logger.warning(f"Empty quote response for {symbol}")
        except Exception as e:
            self.logger.warning(f"Error fetching index price for {symbol}: {str(e)}")
            # Check if it's an invalid API key error
            if "Invalid API key" in str(e):
                self.logger.error(f"Invalid API key detected. Index data may not be included in your subscription plan.")
                # Fall back to mock data immediately
                return self._get_mock_price(symbol)
        
        # Fall back to mock data if we get here
        self.logger.info(f"Falling back to mock data for {symbol}")
        return self._get_mock_price(symbol)
    
    def _get_mock_price(self, symbol):
        """Generate realistic mock price data"""
        # Base prices for different symbols
        base_prices = {
            'ETHUSD': 2200.0,
            'SOLUSD': 170.0,
            'BTCUSD': 42000.0,
            'ADAUSD': 0.45,
            'DOTUSD': 5.80,
            'LINKUSD': 8.20,
            'GBPUSD': 1.2650,
            'USDJPY': 156.50,
            'EURUSD': 1.0820,
            'VIX': 16.5
        }
        
        # Realistic volatility for each symbol
        volatility = {
            'ETHUSD': 0.0015,
            'SOLUSD': 0.0020,
            'BTCUSD': 0.0012,
            'ADAUSD': 0.0025,
            'DOTUSD': 0.0022,
            'LINKUSD': 0.0020,
            'GBPUSD': 0.0003,
            'USDJPY': 0.0004,
            'EURUSD': 0.0002,
            'VIX': 0.0025,
        }
        
        # Get base price and volatility for this symbol
        base_price = base_prices.get(symbol, 100.0)
        vol = volatility.get(symbol, 0.001)
        
        # Generate random price with normal distribution
        random_change = np.random.normal(0, vol)
        mock_price = base_price * (1 + random_change)
        
        # Create result with current timestamp
        result = {
            'price': mock_price,
            'volume': np.random.uniform(100, 10000),
            'timestamp': int(time.time() * 1000),
            'source': 'mock'
        }
        
        # Update the cache
        provider_symbol = self._map_to_provider_symbol(symbol)
        self.last_prices[provider_symbol] = result
        
        return result
    
    def _get_crypto_data(self, symbol, resolution, from_time, to_time):
        """
        Fetch cryptocurrency data
        
        Args:
            symbol: Crypto symbol (e.g., 'ETHUSD', 'SOLUSD')
            resolution: Time resolution
            from_time: Start timestamp
            to_time: End timestamp
            
        Returns:
            Dictionary with OHLCV data
        """
        # Always return mock data for demo key
        if self.api_key == "demo":
            self.logger.debug(f"Using mock crypto data for {symbol} (demo mode)")
            return self._get_mock_crypto_data(symbol, resolution, from_time, to_time)
        
        try:
            # First try using the alternative provider if available
            if self.alt_provider:
                try:
                    alt_data = self.alt_provider.get_crypto_history(symbol, resolution, from_time, to_time)
                    if alt_data and not alt_data.empty:
                        self.logger.debug(f"Successfully fetched {symbol} data from alternative provider")
                        
                        # Convert to Finnhub format
                        result = {
                            's': 'ok',
                            't': alt_data.index.astype(int) // 10**9,
                            'o': alt_data['open'].tolist(),
                            'h': alt_data['high'].tolist(),
                            'l': alt_data['low'].tolist(),
                            'c': alt_data['close'].tolist(),
                            'v': alt_data['volume'].tolist()
                        }
                        return result
                except Exception as e:
                    self.logger.warning(f"Error fetching crypto data from alternative provider for {symbol}: {str(e)}")
            
            # Map to the correct Binance symbol for Finnhub
            binance_symbol_map = {
                'ETHUSD': 'ETHUSDT',
                'SOLUSD': 'SOLUSDT',
                'BTCUSD': 'BTCUSDT', 
                'ADAUSD': 'ADAUSDT',
                'DOTUSD': 'DOTUSDT',
                'LINKUSD': 'LINKUSDT'
            }
            binance_symbol = binance_symbol_map.get(symbol, 'ETHUSDT')
            finnhub_symbol = f'BINANCE:{binance_symbol}'
            
            # Ensure from_time is not too far back (Binance/Finnhub limitation)
            max_periods = {
                '1': 1000,    # 1min: ~16 hours
                '5': 1000,    # 5min: ~3.5 days
                '15': 1000,   # 15min: ~10 days
                '30': 1000,   # 30min: ~20 days
                '60': 1000,   # 1hour: ~41 days
                '240': 1000,  # 4hour: ~166 days
                'D': 1000,    # 1day: ~2.7 years
                'W': 1000,    # 1week: ~19 years
                'M': 1000     # 1month: ~83 years
            }
            
            # Calculate minimum from_time based on resolution
            if resolution in max_periods:
                min_from_time = to_time
                if resolution.isdigit():
                    # For minute-based resolutions
                    min_from_time -= int(resolution) * 60 * max_periods[resolution]
                elif resolution == 'D':
                    # For day resolution
                    min_from_time -= 86400 * max_periods[resolution]
                elif resolution == 'W':
                    # For week resolution
                    min_from_time -= 7 * 86400 * max_periods[resolution]
                elif resolution == 'M':
                    # For month resolution (approximation)
                    min_from_time -= 30 * 86400 * max_periods[resolution]
                
                # Use the later of calculated min_from_time or requested from_time
                from_time = max(from_time, min_from_time)
            
            # Get crypto data from Finnhub
            crypto_data = self.finnhub_client.crypto_candles(
                symbol=finnhub_symbol,
                resolution=resolution,
                _from=from_time,
                to=to_time
            )
            
            if crypto_data and crypto_data['s'] == 'ok' and len(crypto_data['c']) > 0:
                self.logger.debug(f"Successfully fetched {symbol} data from Finnhub")
                return crypto_data
            else:
                if crypto_data:
                    self.logger.warning(f"Invalid crypto data for {symbol}: {crypto_data}")
                else:
                    self.logger.warning(f"Empty crypto data response for {symbol}")
        except Exception as e:
            self.logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
        
        # Fall back to mock data if all methods fail
        self.logger.info(f"Falling back to mock data for {symbol}")
        return self._get_mock_crypto_data(symbol, resolution, from_time, to_time)
            
    def _get_mock_crypto_data(self, symbol, resolution, from_time, to_time):
        """
        Generate mock cryptocurrency data for testing
        
        Args:
            symbol: Crypto symbol (e.g., 'ETHUSD', 'SOLUSD')
            resolution: Time resolution
            from_time: Start timestamp
            to_time: End timestamp
            
        Returns:
            Dictionary with OHLCV data
        """
        # Number of candles to generate
        if resolution.isdigit():
            num_candles = min(100, (to_time - from_time) // (int(resolution) * 60))
        else:
            # D, W, M resolutions
            if resolution == 'D':
                num_candles = min(100, (to_time - from_time) // 86400)
            elif resolution == 'W':
                num_candles = min(100, (to_time - from_time) // (86400 * 7))
            else:  # M
                num_candles = min(100, (to_time - from_time) // (86400 * 30))
                
        # Base prices for different cryptocurrencies
        base_prices = {
            'ETHUSD': 2200.0,
            'SOLUSD': 170.0
        }
        
        # Volatility for different cryptocurrencies
        volatility = {
            'ETHUSD': 50.0,
            'SOLUSD': 15.0
        }
        
        # Generate timestamps
        timestamps = []
        if resolution.isdigit():
            step = int(resolution) * 60
        elif resolution == 'D':
            step = 86400
        elif resolution == 'W':
            step = 86400 * 7
        else:  # M
            step = 86400 * 30
            
        for i in range(num_candles):
            timestamps.append(to_time - (num_candles - i - 1) * step)
            
        # Generate price data with some randomness but realistic movement
        base_price = base_prices.get(symbol, 100.0)
        vol = volatility.get(symbol, 10.0)
        
        # Lists for OHLCV data
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = base_price
        
        for i in range(num_candles):
            # Random price change with a slight upward bias
            change_pct = np.random.normal(0.0002, 0.015)  # Mean slightly positive
            
            # Calculate OHLC
            open_price = current_price
            close_price = open_price * (1 + change_pct)
            
            # High and low with random ranges
            high_range = np.random.uniform(0.003, 0.02)
            low_range = np.random.uniform(0.003, 0.02)
            
            high_price = max(open_price, close_price) * (1 + high_range)
            low_price = min(open_price, close_price) * (1 - low_range)
            
            # Volume with some randomness
            volume = np.random.uniform(100, 10000) * base_price / 1000
            
            # Add to lists
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            # Update current price for next candle
            current_price = close_price
        
        # Return in Finnhub format
        return {
            's': 'ok',
            't': timestamps,
            'o': opens,
            'h': highs,
            'l': lows,
            'c': closes,
            'v': volumes
        }
    
    def get_news(self, category="forex", min_id=0, count=10, use_cache=True):
        """
        Fetch news related to forex or general financial markets
        
        Args:
            category: News category ('forex', 'general', 'economic', 'crypto')
            min_id: Minimum news ID to fetch from
            count: Maximum number of news items to return
            use_cache: Whether to use cache for this request
        """
        # Create cache parameters
        cache_params = {
            'type': 'news',
            'category': category,
            'min_id': min_id
        }
        
        cache_key = self._get_cache_key(cache_params)
        
        # Check cache
        if use_cache and self.use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data[:count]
        
        try:
            # Apply rate limiting
            self._rate_limit_request()
            
            # Map our categories to Finnhub's categories
            category_map = {
                'forex': 'forex',
                'general': 'general',
                'economic': 'general',
                'crypto': 'crypto'
            }
            
            finnhub_category = category_map.get(category, 'general')
            news = self.finnhub_client.general_news(finnhub_category, min_id=min_id)
            
            # Save to cache
            if use_cache and self.use_cache and news:
                self._save_to_cache(cache_key, news)
                
            # Return limited number of items
            return news[:count] if news else []
            
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
    
    def get_economic_calendar(self):
        """Get upcoming economic events"""
        try:
            # Get dates for current week
            now = datetime.datetime.now()
            start_date = now.strftime('%Y-%m-%d')
            end_date = (now + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            
            calendar = self.finnhub_client.economic_calendar(
                _from=start_date,
                to=end_date
            )
            return calendar
        except Exception as e:
            print(f"Error fetching economic calendar: {e}")
            return []
            
    def get_crypto_symbols(self):
        """Get a list of available cryptocurrency symbols"""
        try:
            # Only return the crypto symbols we support
            supported_symbols = {
                'ETHUSD': 'Ethereum / US Dollar',
                'SOLUSD': 'Solana / US Dollar',
                'BTCUSD': 'Bitcoin / US Dollar',
                'ADAUSD': 'Cardano / US Dollar',
                'DOTUSD': 'Polkadot / US Dollar',
                'LINKUSD': 'Chainlink / US Dollar'
            }
            
            return [
                {'symbol': symbol, 'description': description}
                for symbol, description in supported_symbols.items()
            ]
        except Exception as e:
            print(f"Error fetching crypto symbols: {e}")
            return []
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to a dataframe
        
        Args:
            df: Pandas DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        if df.empty:
            return df
            
        # Copy to avoid modifying original
        df_with_indicators = df.copy()
        
        # VWAP
        try:
            df_with_indicators['vwap'] = ta.vwap(df_with_indicators['high'], 
                                              df_with_indicators['low'], 
                                              df_with_indicators['close'], 
                                              df_with_indicators['volume'])
        except:
            df_with_indicators['vwap'] = np.nan
        
        # Bollinger Bands
        try:
            bollinger = ta.bbands(df_with_indicators['close'], length=20, std=2)
            df_with_indicators = pd.concat([df_with_indicators, bollinger], axis=1)
        except:
            pass
        
        # RSI
        try:
            df_with_indicators['rsi'] = ta.rsi(df_with_indicators['close'], length=14)
        except:
            df_with_indicators['rsi'] = np.nan
        
        # ATR for volatility
        try:
            df_with_indicators['atr'] = ta.atr(df_with_indicators['high'], 
                                           df_with_indicators['low'], 
                                           df_with_indicators['close'], 
                                           length=14)
        except:
            df_with_indicators['atr'] = np.nan
            
        # Momentum
        try:
            df_with_indicators['momentum'] = ta.mom(df_with_indicators['close'], length=10)
        except:
            df_with_indicators['momentum'] = np.nan
        
        # Moving averages for trend
        try:
            df_with_indicators['sma_20'] = ta.sma(df_with_indicators['close'], length=20)
            df_with_indicators['sma_50'] = ta.sma(df_with_indicators['close'], length=50)
            df_with_indicators['sma_200'] = ta.sma(df_with_indicators['close'], length=200)
        except:
            df_with_indicators['sma_20'] = np.nan
            df_with_indicators['sma_50'] = np.nan
            df_with_indicators['sma_200'] = np.nan
            
        # MACD
        try:
            macd = ta.macd(df_with_indicators['close'])
            df_with_indicators = pd.concat([df_with_indicators, macd], axis=1)
        except:
            pass
        
        return df_with_indicators
    
    def get_vix_data(self, days=30):
        """
        Fetch VIX index data
        
        Args:
            days: Number of days of data to fetch
            
        Returns:
            DataFrame with VIX data
        """
        try:
            to_date = datetime.datetime.now()
            from_date = to_date - datetime.timedelta(days=days)
            
            to_timestamp = int(to_date.timestamp())
            from_timestamp = int(from_date.timestamp())
            
            # Use Finnhub to get VIX data (symbol: ^VIX)
            data = self.finnhub_client.stock_candles(
                symbol='VIX', 
                resolution='D', 
                _from=from_timestamp, 
                to=to_timestamp
            )
            
            if data['s'] == 'ok':
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                
                df.set_index('timestamp', inplace=True)
                return df
            else:
                print("Error fetching VIX data")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return pd.DataFrame()
    
    def mock_order_flow(self, symbol, length=10):
        """
        Create mock order flow data (since real order book data requires premium APIs)
        
        Args:
            symbol: Forex pair symbol
            length: Length of mock data to generate
            
        Returns:
            DataFrame with mocked order flow
        """
        now = datetime.datetime.now(pytz.UTC)
        dates = [now - datetime.timedelta(minutes=i) for i in range(length)]
        
        # Generate plausible but randomized order flow data
        buy_volume = np.random.normal(1000, 200, length)
        sell_volume = np.random.normal(1000, 200, length)
        
        # Create imbalance
        if symbol == 'GBPUSD':
            buy_imbalance = np.random.choice([1.2, 0.8], size=length, p=[0.6, 0.4])
        elif symbol == 'USDJPY':
            buy_imbalance = np.random.choice([1.3, 0.7], size=length, p=[0.5, 0.5])
        else:
            buy_imbalance = np.random.choice([1.1, 0.9], size=length, p=[0.5, 0.5])
            
        buy_volume = buy_volume * buy_imbalance
        
        df = pd.DataFrame({
            'timestamp': dates,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'net_flow': buy_volume - sell_volume,
            'buy_orders': np.random.randint(50, 150, length),
            'sell_orders': np.random.randint(50, 150, length)
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_yield_curve_data(self):
        """Get US Treasury yield curve data"""
        try:
            # Mock yield curve data since Finnhub free tier doesn't provide this directly
            current_date = datetime.datetime.now()
            
            yields = {
                '1M': 4.5 + np.random.normal(0, 0.05),
                '3M': 4.6 + np.random.normal(0, 0.05),
                '6M': 4.7 + np.random.normal(0, 0.05),
                '1Y': 4.8 + np.random.normal(0, 0.05),
                '2Y': 4.85 + np.random.normal(0, 0.05),
                '5Y': 4.9 + np.random.normal(0, 0.05),
                '10Y': 5.0 + np.random.normal(0, 0.05),
                '30Y': 5.1 + np.random.normal(0, 0.05),
            }
            
            # Calculate simple inversion metrics
            yields['10Y_2Y_spread'] = yields['10Y'] - yields['2Y']
            yields['10Y_3M_spread'] = yields['10Y'] - yields['3M']
            
            return yields
            
        except Exception as e:
            print(f"Error creating yield curve data: {e}")
            return {}
    
    def send_acknowledgment(self, symbol=None):
        """
        Send a ping acknowledgment to keep the WebSocket connection alive
        
        Args:
            symbol: Optional symbol to include in the ping (for debugging)
        
        Returns:
            bool: Whether the acknowledgment was sent successfully
        """
        if not hasattr(self, 'ws') or not self.ws:
            self.logger.debug("No WebSocket connection to acknowledge")
            return False
            
        try:
            if self.ws and hasattr(self.ws, 'sock') and self.ws.sock and hasattr(self.ws.sock, 'connected') and self.ws.sock.connected:
                ping_data = {"type": "ping"}
                if symbol:
                    ping_data["symbol"] = symbol
                
                self.ws.send(json.dumps(ping_data))
                self.logger.debug(f"Sent ping acknowledgment{' for '+symbol if symbol else ''}")
                return True
            else:
                self.logger.debug("WebSocket not connected, can't send acknowledgment")
                return False
        except websocket._exceptions.WebSocketConnectionClosedException:
            self.logger.warning("Can't send acknowledgment - WebSocket connection closed")
            # Don't reconnect here, let the on_close handler handle it
            return False
        except Exception as e:
            self.logger.error(f"Error sending acknowledgment: {e}")
            return False 