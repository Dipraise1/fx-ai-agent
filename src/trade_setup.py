import datetime

class TradeSetup:
    """Represents a trading opportunity with entry, exit, and risk parameters"""
    
    def __init__(self, 
                instrument=None, 
                direction=None, 
                entry_zone=None, 
                stop_loss=None, 
                targets=None, 
                risk_reward=None, 
                confidence=None, 
                rationale=None,
                strategy=None,
                timestamp=None):
        """
        Initialize a trade setup
        
        Args:
            instrument (str): Trading instrument (e.g., 'GBPUSD')
            direction (str): Trade direction ('Long' or 'Short')
            entry_zone (tuple): (min, max) for entry zone
            stop_loss (float): Stop loss level
            targets (list): List of price targets [target1, target2,...]
            risk_reward (float): Risk/reward ratio
            confidence (int): Confidence level (0-100%)
            rationale (str): Reasoning behind the trade
            strategy (str): Strategy used ('GBPUSD_Scalping', 'JPY_Event', 'VIX_Volatility')
            timestamp (datetime): When this setup was generated
        """
        self.instrument = instrument
        self.direction = direction
        self.entry_zone = entry_zone
        self.stop_loss = stop_loss
        self.targets = targets if targets else []
        self.risk_reward = risk_reward
        self.confidence = confidence or 0
        self.rationale = rationale or "No rationale provided"
        self.strategy = strategy
        self.timestamp = timestamp or datetime.datetime.now()
    
    def to_dict(self):
        """Convert trade setup to dictionary for serialization"""
        return {
            'instrument': self.instrument,
            'direction': self.direction,
            'entry_zone': self.entry_zone,
            'stop_loss': self.stop_loss,
            'targets': self.targets,
            'risk_reward': self.risk_reward,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self):
        """Convert trade setup to formatted JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self):
        """String representation of trade setup"""
        # Format entry zone if available
        if self.entry_zone and isinstance(self.entry_zone, (list, tuple)) and len(self.entry_zone) == 2:
            entry_str = f"{self.entry_zone[0]:.5f}-{self.entry_zone[1]:.5f}"
        else:
            entry_str = "N/A"
        
        # Format targets if available
        if self.targets:
            try:
                targets_str = ", ".join([f"{t:.5f}" for t in self.targets])
            except (ValueError, TypeError):
                targets_str = "Invalid targets format"
        else:
            targets_str = "N/A"
        
        # Format stop loss
        stop_loss_str = f"{self.stop_loss:.5f}" if self.stop_loss is not None else "N/A"
        
        # Format risk/reward
        risk_reward_str = f"{self.risk_reward:.2f}" if self.risk_reward is not None else "N/A"
        
        # Build the string
        setup_str = f"TRADE SETUP: {self.instrument} - {self.strategy}\n"
        setup_str += f"Direction: {self.direction}\n"
        setup_str += f"Entry Zone: {entry_str}\n"
        setup_str += f"Stop Loss: {stop_loss_str}\n"
        setup_str += f"Targets: {targets_str}\n"
        setup_str += f"Risk/Reward: {risk_reward_str}\n"
        setup_str += f"Confidence: {self.confidence}%\n"
        setup_str += f"Rationale: {self.rationale}\n"
        setup_str += f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return setup_str
    
    @staticmethod
    def no_trade(instrument, strategy, reason):
        """Create a no-trade setup with explanation"""
        return TradeSetup(
            instrument=instrument,
            direction="No Trade",
            strategy=strategy,
            rationale=reason,
            confidence=0
        )
        
    @staticmethod
    def calculate_risk_reward(entry, stop, target):
        """Calculate risk/reward ratio"""
        if not (entry and stop and target):
            return None
            
        if isinstance(entry, tuple):
            entry = sum(entry) / 2  # Use middle of entry zone
            
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if risk == 0:
            return None
            
        return reward / risk 