"""
Risk management module for trading operations.
Handles position sizing, portfolio risk, and exposure calculations.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_risk: float = 0.01):
        """
        Initialize the risk manager.
        
        Args:
            max_portfolio_risk: Maximum allowed portfolio risk (default: 2%)
            max_position_risk: Maximum allowed risk per position (default: 1%)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        risk_per_trade: Optional[float] = None
    ) -> Tuple[int, float]:
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            capital: Available trading capital
            entry_price: Planned entry price
            stop_loss: Stop loss price
            risk_per_trade: Risk amount per trade (defaults to max_position_risk)
            
        Returns:
            Tuple of (position size in shares, dollar risk)
        """
        if risk_per_trade is None:
            risk_per_trade = self.max_position_risk
            
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            raise ValueError("Entry price cannot equal stop loss price")
            
        # Calculate maximum dollar risk
        max_risk = capital * risk_per_trade
        
        # Calculate position size
        position_size = int(max_risk / risk_per_share)
        actual_risk = position_size * risk_per_share
        
        return position_size, actual_risk
        
    def calculate_portfolio_risk(
        self,
        positions: List[Dict[str, float]],
        correlations: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate total portfolio risk considering position correlations.
        
        Args:
            positions: List of position dictionaries with 'weight' and 'volatility'
            correlations: Optional correlation matrix between positions
            
        Returns:
            Portfolio risk as a decimal
        """
        if not positions:
            return 0.0
            
        weights = np.array([p['weight'] for p in positions])
        vols = np.array([p['volatility'] for p in positions])
        
        if correlations is None:
            # Assume no correlation between positions
            correlations = pd.DataFrame(
                np.identity(len(positions)),
                index=[p['symbol'] for p in positions],
                columns=[p['symbol'] for p in positions]
            )
            
        # Calculate covariance matrix
        cov_matrix = np.outer(vols, vols) * correlations
        
        # Calculate portfolio variance
        portfolio_variance = weights.dot(cov_matrix).dot(weights)
        
        return np.sqrt(portfolio_variance)
        
    def calculate_sector_exposure(self, positions: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate exposure by sector.
        
        Args:
            positions: List of position dictionaries with 'weight' and 'sector'
            
        Returns:
            Dictionary of sector exposures
        """
        sector_exposure = {}
        for position in positions:
            sector = position['sector']
            weight = position['weight']
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
            
        return sector_exposure
        
    def calculate_portfolio_correlation(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate portfolio correlation with a benchmark.
        
        Args:
            returns: DataFrame of position returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Correlation coefficient
        """
        portfolio_returns = returns.mean(axis=1)  # Equally weighted portfolio
        return portfolio_returns.corr(benchmark_returns)
        
    def validate_execution(
        self,
        order_size: int,
        capital: float,
        current_positions: List[Dict[str, float]]
    ) -> bool:
        """
        Validate if an order execution would violate risk limits.
        
        Args:
            order_size: Size of the order in shares
            capital: Available capital
            current_positions: List of current position dictionaries
            
        Returns:
            True if execution is valid, False otherwise
        """
        # Calculate current portfolio risk
        current_risk = self.calculate_portfolio_risk(current_positions)
        
        # Check if new order would exceed max portfolio risk
        if current_risk > self.max_portfolio_risk:
            return False
            
        # Check if order size would exceed max position risk
        position_risk = order_size / capital
        if position_risk > self.max_position_risk:
            return False
            
        return True
