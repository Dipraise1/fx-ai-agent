#!/usr/bin/env python3
import os
import argparse
import time
import signal
import sys
from dotenv import load_dotenv

# Import NLTK setup utility for fallbacks (do this first)
from src.utils.nltk_setup import setup_fallbacks

# Set up NLTK fallbacks before importing other modules
setup_fallbacks()

# Now import the rest of the modules
from src.agent import TradingAgent
from src.data_fetcher import DataFetcher
from src.utils.logger import Logger

# Global variables
running = True
agent = None

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) to gracefully shut down"""
    global running
    print("\nShutting down trading agent...\n")
    running = False
    if agent and agent.real_time_mode:
        agent.stop_real_time_monitoring()
    sys.exit(0)

def run_realtime_agent(demo_mode=False):
    """
    Run the trading agent in real-time mode with Finnhub data
    
    Args:
        demo_mode: Whether to run in demo mode without API credentials
    """
    print("\n" + "="*80)
    print(" "*30 + "REAL-TIME TRADING AGENT")
    print("="*80 + "\n")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize logger
    logger = Logger()
    logger.info("Starting real-time trading agent")
    
    # Get API credentials
    api_key = "demo" if demo_mode else os.getenv("FINNHUB_API_KEY")
    api_secret = None if demo_mode else os.getenv("FINNHUB_SECRET")
    
    if not api_key and not demo_mode:
        logger.error("Finnhub API key not found. Run with --demo flag or set FINNHUB_API_KEY in .env")
        return
    
    if not api_secret and not demo_mode:
        logger.warning("Finnhub secret not found. Set FINNHUB_SECRET in .env for proper authentication")
    
    # Get real-time settings
    update_interval = int(os.getenv("UPDATE_INTERVAL", "60"))
    use_cache = os.getenv("USE_CACHE", "true").lower() == "true"
    cache_expiry = int(os.getenv("CACHE_EXPIRY", "3600"))
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(
        api_key=api_key, 
        api_secret=api_secret,
        use_cache=use_cache,
        cache_expiry=cache_expiry
    )
    
    # Initialize trading agent in real-time mode
    agent = TradingAgent(
        data_fetcher=data_fetcher,
        real_time_mode=True,
        update_interval=update_interval
    )
    
    # Run initial market analysis
    logger.info("Running initial market analysis...")
    agent.analyze_markets()
    
    print("\n" + "="*80)
    print(" "*25 + "REAL-TIME MONITORING ACTIVE")
    print("="*80 + "\n")
    
    try:
        # Keep the main thread running to allow background threads to work
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle clean shutdown
        logger.info("Stopping real-time monitoring...")
        agent.stop_real_time_monitoring()
        print("\nReal-time monitoring stopped. Exiting...")

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Run trading agent in real-time mode")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without API credentials")
    args = parser.parse_args()
    
    run_realtime_agent(demo_mode=args.demo) 