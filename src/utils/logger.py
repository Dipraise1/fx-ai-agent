import logging
import os
import datetime

class Logger:
    """Simple logging utility for the trading agent"""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize logger with specified log level"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('trading_agent')
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create file handler
        log_file = f"logs/trading_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_trade_setup(self, trade_setup):
        """Log trade setup details"""
        self.info(f"Trade Setup Generated: {trade_setup.instrument} - {trade_setup.strategy}")
        self.info(str(trade_setup))
    
    def log_market_commentary(self, commentary):
        """Log market commentary"""
        self.info("Market Commentary:")
        self.info(commentary) 