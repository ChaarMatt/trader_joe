"""
Trade execution module for handling order placement and management.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import numpy as np

from src.analysis.risk_manager import RiskManager
from src.data.market_data import MarketDataManager as MarketData
from src.data.broker_client import BrokerClient

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(
        self,
        market_data: MarketData,
        broker_client: BrokerClient,
        risk_manager: Optional[RiskManager] = None
    ):
        """
        Initialize the trade executor.
        
        Args:
            market_data: Market data provider instance
            broker_client: Broker client instance
            risk_manager: Optional risk manager instance
        """
        self.market_data = market_data
        self.broker_client = broker_client
        self.risk_manager = risk_manager or RiskManager()
        
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict:
        """
        Execute a trade with the specified parameters.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            limit_price: Optional limit price for limit orders
            stop_price: Optional stop price for stop orders
            
        Returns:
            Order execution details
        """
        try:
            # Get current market data
            quote = await self.market_data.get_real_time_quote(symbol)
            
            # Validate trade parameters
            if not self._validate_trade_params(side, quantity, order_type, limit_price, stop_price):
                raise ValueError("Invalid trade parameters")
                
            # Check risk limits
            if not await self._check_risk_limits(symbol, side, quantity):
                raise ValueError("Trade exceeds risk limits")
                
            # Place order
            order = await self.broker_client.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            logger.info(f"Executed {side} order for {quantity} shares of {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {str(e)}")
            raise
            
    def _validate_trade_params(
        self,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float],
        stop_price: Optional[float]
    ) -> bool:
        """Validate trade parameters."""
        # Check side
        if side not in ['buy', 'sell']:
            return False
            
        # Check quantity
        if quantity <= 0:
            return False
            
        # Check order type and prices
        if order_type not in ['market', 'limit', 'stop', 'stop_limit']:
            return False
            
        if order_type in ['limit', 'stop_limit'] and limit_price is None:
            return False
            
        if order_type in ['stop', 'stop_limit'] and stop_price is None:
            return False
            
        return True
        
    async def _check_risk_limits(self, symbol: str, side: str, quantity: int) -> bool:
        """Check if trade complies with risk limits."""
        try:
            # Get current positions
            positions = await self.broker_client.get_positions()
            
            # Get account info
            account = await self.broker_client.get_account()
            capital = float(account['buying_power'])
            
            # Validate execution
            return self.risk_manager.validate_execution(
                order_size=quantity,
                capital=capital,
                current_positions=[
                    {
                        'symbol': p['symbol'],
                        'weight': float(p['market_value']) / capital,
                        'volatility': await self._get_volatility(p['symbol'])
                    }
                    for p in positions
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to check risk limits: {str(e)}")
            return False
            
    async def _get_volatility(self, symbol: str, lookback_days: int = 30) -> float:
        """Calculate historical volatility for a symbol."""
        try:
            # Get historical data
            history = await self.market_data.get_market_data(
                symbol,
                period=f"{lookback_days}d",
                interval="1d"
            )
            
            # Check if we got valid data
            if history is None or history.empty:
                logger.warning(f"No historical data available for {symbol}")
                return 0.0
                
            # Calculate daily returns
            returns = history['Close'].pct_change().dropna()  # Note: MarketDataManager uses 'Close' not 'close'
            
            # Calculate annualized volatility
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            return annual_vol
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility for {symbol}: {str(e)}")
            return 0.0
