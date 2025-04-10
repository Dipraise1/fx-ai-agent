import numpy as np
import pandas as pd
import datetime
import logging
import os
import json

class RiskManager:
    """Risk management model for trading"""
    
    def __init__(self, account_size=10000, max_risk_percent=1.0, max_daily_loss=3.0, max_positions=3):
        """
        Initialize the risk management model
        
        Args:
            account_size (float): Starting account size in base currency
            max_risk_percent (float): Maximum risk per trade as percent of account
            max_daily_loss (float): Maximum allowed daily loss as percent of account
            max_positions (int): Maximum number of concurrent positions
        """
        self.account_size = account_size
        self.max_risk_percent = max_risk_percent
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        
        # Current account state
        self.current_balance = account_size
        self.open_positions = []
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.datetime.now().date()
        
        # Performance metrics
        self.realized_trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.break_even_trades = 0
        
        # Journal file
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        self.journal_file = os.path.join(self.log_dir, 'trade_journal.json')
        
        # Load existing journal if available
        self._load_journal()
        
        # Logger
        self.logger = logging.getLogger('trading_agent.risk_management')
    
    def _load_journal(self):
        """Load trade journal from file if it exists"""
        try:
            if os.path.exists(self.journal_file):
                with open(self.journal_file, 'r') as f:
                    journal_data = json.load(f)
                    
                self.current_balance = journal_data.get('current_balance', self.account_size)
                self.realized_trades = journal_data.get('realized_trades', [])
                self.total_trades = journal_data.get('total_trades', 0)
                self.winning_trades = journal_data.get('winning_trades', 0)
                self.losing_trades = journal_data.get('losing_trades', 0)
                self.break_even_trades = journal_data.get('break_even_trades', 0)
                
                # Convert string date back to datetime.date
                last_reset_str = journal_data.get('last_reset_date', None)
                if last_reset_str:
                    self.last_reset_date = datetime.datetime.strptime(
                        last_reset_str, '%Y-%m-%d'
                    ).date()
                
                # Current day's PnL
                if self.last_reset_date == datetime.datetime.now().date():
                    self.daily_pnl = journal_data.get('daily_pnl', 0.0)
                else:
                    # Reset daily PnL if it's a new day
                    self.daily_pnl = 0.0
                    self.last_reset_date = datetime.datetime.now().date()
                
                # Open positions
                self.open_positions = journal_data.get('open_positions', [])
                
                self.logger.info(f"Loaded trade journal: {len(self.realized_trades)} historical trades")
        except Exception as e:
            self.logger.error(f"Error loading trade journal: {e}")
    
    def _save_journal(self):
        """Save trade journal to file"""
        try:
            journal_data = {
                'current_balance': self.current_balance,
                'open_positions': self.open_positions,
                'daily_pnl': self.daily_pnl,
                'last_reset_date': self.last_reset_date.strftime('%Y-%m-%d'),
                'realized_trades': self.realized_trades,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'break_even_trades': self.break_even_trades
            }
            
            with open(self.journal_file, 'w') as f:
                json.dump(journal_data, f, indent=2)
                
            self.logger.info(f"Trade journal saved to {self.journal_file}")
        except Exception as e:
            self.logger.error(f"Error saving trade journal: {e}")
    
    def calculate_position_size(self, trade_setup, current_price=None):
        """
        Calculate appropriate position size based on risk parameters
        
        Args:
            trade_setup: TradeSetup object with trade details
            current_price (float): Current market price (optional)
            
        Returns:
            dict: Position size information
        """
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss * self.account_size / 100:
            self.logger.warning("Daily loss limit reached, no new positions allowed")
            return {
                "position_allowed": False,
                "reason": "Daily loss limit reached",
                "max_position_size": 0,
                "recommended_position_size": 0,
                "risk_amount": 0,
                "risk_percent": 0
            }
            
        # Check maximum positions
        if len(self.open_positions) >= self.max_positions:
            self.logger.warning("Maximum number of positions reached")
            return {
                "position_allowed": False,
                "reason": "Maximum number of positions reached",
                "max_position_size": 0,
                "recommended_position_size": 0,
                "risk_amount": 0,
                "risk_percent": 0
            }
        
        # Get entry and stop prices
        if trade_setup.entry_zone and trade_setup.stop_loss:
            # Use middle of entry zone
            if isinstance(trade_setup.entry_zone, tuple):
                entry_price = sum(trade_setup.entry_zone) / 2
            else:
                entry_price = trade_setup.entry_zone
                
            stop_price = trade_setup.stop_loss
        else:
            # If no entry or stop defined, cannot calculate position size
            self.logger.warning("No entry zone or stop loss defined in trade setup")
            return {
                "position_allowed": False,
                "reason": "No entry zone or stop loss defined",
                "max_position_size": 0,
                "recommended_position_size": 0,
                "risk_amount": 0,
                "risk_percent": 0
            }
            
        # Override entry with current price if provided
        if current_price:
            entry_price = current_price
        
        # Calculate risk per unit
        if trade_setup.direction == "Long":
            risk_per_unit = entry_price - stop_price
        else:  # "Short"
            risk_per_unit = stop_price - entry_price
            
        # Sanity check
        if risk_per_unit <= 0:
            self.logger.warning("Invalid risk calculation: Stop loss should be below entry for longs, above for shorts")
            return {
                "position_allowed": False,
                "reason": "Invalid stop loss placement",
                "max_position_size": 0,
                "recommended_position_size": 0,
                "risk_amount": 0,
                "risk_percent": 0
            }
            
        # Calculate maximum risk amount in currency
        max_risk = self.current_balance * self.max_risk_percent / 100
        
        # Adjust based on confidence level
        confidence_factor = trade_setup.confidence / 100 if trade_setup.confidence else 0.5
        adjusted_risk = max_risk * confidence_factor
        
        # Calculate position size
        max_position_size = max_risk / risk_per_unit
        recommended_position_size = adjusted_risk / risk_per_unit
        
        # For forex, convert to lot size (standard lot = 100,000 units)
        is_fx = trade_setup.instrument in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        if is_fx:
            # Round down to nearest 0.01 lot (1000 units)
            max_position_size = np.floor(max_position_size / 1000) / 100
            recommended_position_size = np.floor(recommended_position_size / 1000) / 100
        else:
            # For non-FX, round to nearest whole unit
            max_position_size = np.floor(max_position_size)
            recommended_position_size = np.floor(recommended_position_size)
            
        # Ensure minimum position size
        if is_fx and recommended_position_size < 0.01:
            recommended_position_size = 0.01  # Minimum 0.01 lot (micro lot)
        elif not is_fx and recommended_position_size < 1:
            recommended_position_size = 1  # Minimum 1 unit
            
        # Calculate actual risk amount and percent
        actual_risk = recommended_position_size * risk_per_unit
        risk_percent = (actual_risk / self.current_balance) * 100
            
        return {
            "position_allowed": True,
            "max_position_size": max_position_size,
            "recommended_position_size": recommended_position_size,
            "risk_amount": actual_risk,
            "risk_percent": risk_percent,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "confidence_factor": confidence_factor
        }
    
    def open_position(self, trade_setup, position_size=None, entry_price=None, timestamp=None):
        """
        Open a new trading position
        
        Args:
            trade_setup: TradeSetup object with trade details
            position_size (float): Position size to open
            entry_price (float): Actual entry price
            timestamp (datetime): Timestamp of the entry
            
        Returns:
            dict: Position information
        """
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss * self.account_size / 100:
            self.logger.warning("Daily loss limit reached, cannot open new position")
            return None
            
        # Check maximum positions
        if len(self.open_positions) >= self.max_positions:
            self.logger.warning("Maximum number of positions reached")
            return None
            
        # Calculate position size if not provided
        if position_size is None:
            sizing = self.calculate_position_size(trade_setup, current_price=entry_price)
            if not sizing["position_allowed"]:
                return None
            position_size = sizing["recommended_position_size"]
            
        # Use middle of entry zone if entry_price not provided
        if entry_price is None:
            if isinstance(trade_setup.entry_zone, tuple):
                entry_price = sum(trade_setup.entry_zone) / 2
            else:
                entry_price = trade_setup.entry_zone
                
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        # Create position object
        position = {
            "instrument": trade_setup.instrument,
            "direction": trade_setup.direction,
            "entry_price": entry_price,
            "stop_loss": trade_setup.stop_loss,
            "targets": trade_setup.targets,
            "position_size": position_size,
            "open_time": timestamp.isoformat(),
            "strategy": trade_setup.strategy,
            "risk_reward": trade_setup.risk_reward,
            "confidence": trade_setup.confidence,
            "rationale": trade_setup.rationale,
            "status": "open",
            "close_price": None,
            "close_time": None,
            "pnl": 0,
            "pnl_percent": 0,
            "position_id": f"{trade_setup.instrument}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Add position to open positions
        self.open_positions.append(position)
        
        # Log position
        self.logger.info(f"Opened {position['direction']} position: {position['instrument']}, " +
                       f"Size: {position['position_size']}, Entry: {position['entry_price']}")
        
        # Save journal
        self._save_journal()
        
        return position
    
    def close_position(self, position_id, close_price, close_time=None, reason="Target reached"):
        """
        Close an open trading position
        
        Args:
            position_id (str): ID of the position to close
            close_price (float): Price at which position is closed
            close_time (datetime): Time of closure
            reason (str): Reason for closing the position
            
        Returns:
            dict: Closed position information
        """
        # Find position by ID
        position_index = None
        for i, pos in enumerate(self.open_positions):
            if pos["position_id"] == position_id:
                position_index = i
                break
                
        if position_index is None:
            self.logger.warning(f"Position not found: {position_id}")
            return None
            
        # Get position
        position = self.open_positions[position_index]
        
        # Update close information
        if close_time is None:
            close_time = datetime.datetime.now()
            
        position["close_price"] = close_price
        position["close_time"] = close_time.isoformat()
        position["status"] = "closed"
        position["close_reason"] = reason
        
        # Calculate P&L
        if position["direction"] == "Long":
            pnl = (close_price - position["entry_price"]) * position["position_size"]
        else:  # "Short"
            pnl = (position["entry_price"] - close_price) * position["position_size"]
            
        position["pnl"] = pnl
        position["pnl_percent"] = (pnl / self.current_balance) * 100
        
        # Update balance
        self.current_balance += pnl
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Update trade statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
        else:
            self.break_even_trades += 1
            
        # Copy to realized trades and remove from open positions
        self.realized_trades.append(position.copy())
        self.open_positions.pop(position_index)
        
        # Log closure
        self.logger.info(f"Closed {position['direction']} position: {position['instrument']}, " +
                       f"PnL: {pnl:.2f} ({position['pnl_percent']:.2f}%), Reason: {reason}")
        
        # Save journal
        self._save_journal()
        
        return position
    
    def check_positions(self, current_prices):
        """
        Check if any open positions need to be closed based on current prices
        
        Args:
            current_prices (dict): Dictionary of current prices by instrument
            
        Returns:
            list: List of positions that were closed
        """
        closed_positions = []
        
        # Check if daily loss limit is reset (new day)
        today = datetime.datetime.now().date()
        if today > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today
            self.logger.info("New trading day - reset daily PnL")
        
        # Check each open position
        for position in self.open_positions[:]:  # Create a copy to iterate over
            instrument = position["instrument"]
            
            # Skip if no price available for this instrument
            if instrument not in current_prices:
                continue
                
            current_price = current_prices[instrument]
            
            # Check stop loss
            if position["direction"] == "Long" and current_price <= position["stop_loss"]:
                closed_position = self.close_position(
                    position["position_id"], position["stop_loss"], reason="Stop loss hit"
                )
                closed_positions.append(closed_position)
                continue
                
            if position["direction"] == "Short" and current_price >= position["stop_loss"]:
                closed_position = self.close_position(
                    position["position_id"], position["stop_loss"], reason="Stop loss hit"
                )
                closed_positions.append(closed_position)
                continue
                
            # Check targets
            for i, target in enumerate(position["targets"]):
                target_hit = False
                
                if position["direction"] == "Long" and current_price >= target:
                    target_hit = True
                elif position["direction"] == "Short" and current_price <= target:
                    target_hit = True
                    
                if target_hit:
                    closed_position = self.close_position(
                        position["position_id"], target, reason=f"Target {i+1} reached"
                    )
                    closed_positions.append(closed_position)
                    break  # Exit the target loop as position is now closed
        
        return closed_positions
    
    def get_account_status(self):
        """
        Get current account status and metrics
        
        Returns:
            dict: Account status information
        """
        # Calculate performance metrics
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        loss_rate = self.losing_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate average win and loss
        avg_win = 0
        avg_loss = 0
        
        if self.winning_trades > 0:
            winning_trades = [t for t in self.realized_trades if t["pnl"] > 0]
            total_wins = sum(t["pnl"] for t in winning_trades)
            avg_win = total_wins / self.winning_trades
            
        if self.losing_trades > 0:
            losing_trades = [t for t in self.realized_trades if t["pnl"] < 0]
            total_losses = sum(t["pnl"] for t in losing_trades)
            avg_loss = total_losses / self.losing_trades
            
        # Calculate profit factor
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calculate expected value
        expected_value = (win_rate * avg_win) + (loss_rate * avg_loss)
        
        # Calculate drawdown
        max_balance = self.account_size
        max_drawdown = 0
        current_drawdown = 0
        
        running_balance = self.account_size
        for trade in self.realized_trades:
            running_balance += trade["pnl"]
            max_balance = max(max_balance, running_balance)
            current_drawdown = (max_balance - running_balance) / max_balance * 100
            max_drawdown = max(max_drawdown, current_drawdown)
            
        # Calculate ROI
        roi = (self.current_balance - self.account_size) / self.account_size * 100
        
        return {
            "current_balance": self.current_balance,
            "starting_balance": self.account_size,
            "open_positions": len(self.open_positions),
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percent": self.daily_pnl / self.account_size * 100,
            "daily_loss_limit": self.max_daily_loss,
            "max_positions": self.max_positions,
            "max_risk_percent": self.max_risk_percent,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "break_even_trades": self.break_even_trades,
            "win_rate": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expected_value": expected_value,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "roi": roi,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def adjust_risk_for_correlation(self, trade_setups):
        """
        Adjust risk for correlated instruments
        
        Args:
            trade_setups (list): List of trade setups to analyze
            
        Returns:
            list: List of adjusted trade setups
        """
        # If only one setup, no correlation to consider
        if len(trade_setups) <= 1:
            return trade_setups
            
        # Define correlation groups
        fx_majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']
        
        # Check for multiple setups in the same correlation group
        correlated_setups = {}
        for group in [fx_majors]:
            group_setups = [ts for ts in trade_setups if ts.instrument in group]
            if len(group_setups) > 1:
                correlated_setups[tuple(group)] = group_setups
                
        # If no correlations found, return original setups
        if not correlated_setups:
            return trade_setups
            
        # Adjust risk for correlated instruments
        adjusted_setups = []
        for ts in trade_setups:
            # Check if this setup is in a correlated group
            is_correlated = False
            for group, setups in correlated_setups.items():
                if ts.instrument in group and ts in setups:
                    is_correlated = True
                    # Reduce confidence proportionally to the number of correlated setups
                    adjusted_confidence = ts.confidence * (1 - 0.2 * (len(setups) - 1))
                    # Create a new setup with adjusted confidence
                    new_ts = type(ts)(
                        instrument=ts.instrument,
                        direction=ts.direction,
                        entry_zone=ts.entry_zone,
                        stop_loss=ts.stop_loss,
                        targets=ts.targets,
                        risk_reward=ts.risk_reward,
                        confidence=adjusted_confidence,  # Adjusted confidence
                        rationale=ts.rationale + " (Risk adjusted for correlation)",
                        strategy=ts.strategy,
                        timestamp=ts.timestamp
                    )
                    adjusted_setups.append(new_ts)
                    break
                    
            # If not correlated, keep original
            if not is_correlated:
                adjusted_setups.append(ts)
                
        return adjusted_setups
    
    def prepare_daily_report(self):
        """
        Prepare a daily trading report
        
        Returns:
            str: Daily report in text format
        """
        # Get account status
        status = self.get_account_status()
        
        # Format report
        report = "===== DAILY TRADING REPORT =====\n"
        report += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        report += "ACCOUNT SUMMARY:\n"
        report += f"Current Balance: {status['current_balance']:.2f}\n"
        report += f"Daily P&L: {status['daily_pnl']:.2f} ({status['daily_pnl_percent']:.2f}%)\n"
        report += f"Open Positions: {status['open_positions']}\n"
        
        report += "\nPERFORMANCE METRICS:\n"
        report += f"Total Trades: {status['total_trades']}\n"
        report += f"Win Rate: {status['win_rate']:.2f}%\n"
        report += f"Profit Factor: {status['profit_factor']:.2f}\n"
        report += f"Average Win: {status['avg_win']:.2f}\n"
        report += f"Average Loss: {status['avg_loss']:.2f}\n"
        report += f"Expected Value: {status['expected_value']:.2f}\n"
        report += f"Max Drawdown: {status['max_drawdown']:.2f}%\n"
        report += f"ROI: {status['roi']:.2f}%\n"
        
        # Today's trades
        today = datetime.datetime.now().date()
        today_trades = []
        
        for trade in self.realized_trades:
            close_time = datetime.datetime.fromisoformat(trade['close_time'])
            if close_time.date() == today:
                today_trades.append(trade)
                
        if today_trades:
            report += "\nTODAY'S COMPLETED TRADES:\n"
            for trade in today_trades:
                report += f"{trade['instrument']} {trade['direction']}: "
                report += f"P&L: {trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%), "
                report += f"Reason: {trade['close_reason']}\n"
                
        # Open positions
        if self.open_positions:
            report += "\nCURRENT OPEN POSITIONS:\n"
            for pos in self.open_positions:
                report += f"{pos['instrument']} {pos['direction']}: "
                report += f"Entry: {pos['entry_price']}, Stop: {pos['stop_loss']}, "
                report += f"Size: {pos['position_size']}\n"
                
        report += "\n" + "="*30 + "\n"
        
        return report 