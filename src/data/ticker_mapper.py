"""
Ticker mapping functionality for Trader Joe.
Maps company names and entities to stock tickers.
"""

import logging
from typing import Dict, List, Optional, Any
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class TickerMapper:
    """Maps company names and entities to stock tickers using a database."""
    
    def __init__(self, db_path: str = 'data/trader_joe.db'):
        self.db_path = db_path
        self._setup_database()
        
    def _setup_database(self) -> None:
        """Initialize SQLite database and create necessary tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS entities
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     ticker TEXT NOT NULL,
                     confidence REAL NOT NULL,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)')
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise

    async def get_ticker(self, text: str) -> Optional[tuple[str, float]]:
        """Get the ticker for a company mentioned in the text."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all company names from the database
                cursor = conn.execute('SELECT name, ticker, confidence FROM entities')
                mappings = cursor.fetchall()
                
                # Sort by name length (descending) to match longer names first
                mappings.sort(key=lambda x: len(x[0]), reverse=True)
                
                # Look for company names in the text
                text = text.lower()
                for name, ticker, confidence in mappings:
                    if name.lower() in text:
                        return (ticker, confidence)
                
                return None
        except Exception as e:
            logger.error(f"Failed to get ticker from text: {e}")
            return None

    async def get_tickers(self, text: str) -> List[tuple[str, float]]:
        """Get all tickers mentioned in the text."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all company names from the database
                cursor = conn.execute('SELECT name, ticker, confidence FROM entities')
                mappings = cursor.fetchall()
                
                # Sort by name length (descending) to match longer names first
                mappings.sort(key=lambda x: len(x[0]), reverse=True)
                
                # Look for all company names in the text
                text = text.lower()
                found_tickers = []
                for name, ticker, confidence in mappings:
                    if name.lower() in text:
                        found_tickers.append((ticker, confidence))
                
                return list(set(found_tickers))  # Remove duplicates
        except Exception as e:
            logger.error(f"Failed to get tickers from text: {e}")
            return []

if __name__ == '__main__':
    # Example usage
    import asyncio
    
    async def main():
        mapper = TickerMapper()
        ticker = await mapper.get_ticker("Apple announced new iPhone today")
        print(f"Found ticker: {ticker}")
        
        tickers = await mapper.get_tickers("Microsoft and Google announced new AI features")
        print(f"Found tickers: {tickers}")
    
    asyncio.run(main())
