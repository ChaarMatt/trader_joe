"""Database models for Trader Joe."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class WatchlistItem:
    id: Optional[int]
    symbol: str
    added_at: datetime
    notes: Optional[str] = None
    
@dataclass
class PortfolioHolding:
    id: Optional[int]
    symbol: str
    quantity: float
    average_cost: float
    purchase_date: datetime
    notes: Optional[str] = None

def init_db(db_path: str):
    """Initialize database tables."""
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            average_cost REAL NOT NULL,
            purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            target_price REAL NOT NULL,
            is_above BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            triggered BOOLEAN DEFAULT FALSE
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            layout_config TEXT,
            theme TEXT DEFAULT 'dark',
            default_timeframe TEXT DEFAULT '1D',
            notification_settings TEXT
        )
        ''')

def get_watchlist(db_path: str) -> List[WatchlistItem]:
    """Get all watchlist items."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('SELECT * FROM watchlist ORDER BY added_at DESC')
        return [WatchlistItem(**dict(row)) for row in cursor.fetchall()]

def add_to_watchlist(db_path: str, symbol: str, notes: Optional[str] = None) -> WatchlistItem:
    """Add a symbol to watchlist."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            'INSERT INTO watchlist (symbol, notes) VALUES (?, ?)',
            (symbol, notes)
        )
        item_id = cursor.lastrowid
        return WatchlistItem(
            id=item_id,
            symbol=symbol,
            added_at=datetime.now(),
            notes=notes
        )

def remove_from_watchlist(db_path: str, symbol: str) -> bool:
    """Remove a symbol from watchlist."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute('DELETE FROM watchlist WHERE symbol = ?', (symbol,))
        return cursor.rowcount > 0

def get_portfolio(db_path: str) -> List[PortfolioHolding]:
    """Get all portfolio holdings."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('SELECT * FROM portfolio ORDER BY purchase_date DESC')
        return [PortfolioHolding(**dict(row)) for row in cursor.fetchall()]

def add_to_portfolio(
    db_path: str,
    symbol: str,
    quantity: float,
    average_cost: float,
    notes: Optional[str] = None
) -> PortfolioHolding:
    """Add a holding to portfolio."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            '''INSERT INTO portfolio 
               (symbol, quantity, average_cost, notes)
               VALUES (?, ?, ?, ?)''',
            (symbol, quantity, average_cost, notes)
        )
        holding_id = cursor.lastrowid
        return PortfolioHolding(
            id=holding_id,
            symbol=symbol,
            quantity=quantity,
            average_cost=average_cost,
            purchase_date=datetime.now(),
            notes=notes
        )

def update_portfolio_holding(
    db_path: str,
    symbol: str,
    quantity: float,
    average_cost: float
) -> bool:
    """Update an existing portfolio holding."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            '''UPDATE portfolio 
               SET quantity = ?, average_cost = ?
               WHERE symbol = ?''',
            (quantity, average_cost, symbol)
        )
        return cursor.rowcount > 0

def remove_from_portfolio(db_path: str, symbol: str) -> bool:
    """Remove a holding from portfolio."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute('DELETE FROM portfolio WHERE symbol = ?', (symbol,))
        return cursor.rowcount > 0

def set_price_alert(
    db_path: str,
    symbol: str,
    target_price: float,
    is_above: bool
) -> Optional[int]:
    """Set a new price alert."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            '''INSERT INTO price_alerts 
               (symbol, target_price, is_above)
               VALUES (?, ?, ?)''',
            (symbol, target_price, is_above)
        )
        return cursor.lastrowid

def get_active_alerts(db_path: str) -> List[dict]:
    """Get all non-triggered price alerts."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            '''SELECT * FROM price_alerts 
               WHERE triggered = FALSE 
               ORDER BY created_at DESC'''
        )
        return [dict(row) for row in cursor.fetchall()]

def mark_alert_triggered(db_path: str, alert_id: int):
    """Mark a price alert as triggered."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'UPDATE price_alerts SET triggered = TRUE WHERE id = ?',
            (alert_id,)
        )

def get_user_preferences(db_path: str) -> dict:
    """Get user preferences."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('SELECT * FROM user_preferences LIMIT 1')
        row = cursor.fetchone()
        return dict(row) if row else {}

def update_user_preferences(db_path: str, preferences: dict) -> bool:
    """Update user preferences."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            '''INSERT OR REPLACE INTO user_preferences 
               (id, layout_config, theme, default_timeframe, notification_settings)
               VALUES (1, ?, ?, ?, ?)''',
            (
                preferences.get('layout_config'),
                preferences.get('theme', 'dark'),
                preferences.get('default_timeframe', '1D'),
                preferences.get('notification_settings')
            )
        )
        return cursor.rowcount > 0
