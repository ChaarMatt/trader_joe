"""
Initialize ticker database with common stock mappings.
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Common stock ticker mappings with company names and variations
TICKER_MAPPINGS = [
    # Tech companies
    ('AAPL', ['Apple', 'Apple Inc', 'Apple Inc.', 'AAPL'], 1.0),
    ('MSFT', ['Microsoft', 'Microsoft Corporation', 'MSFT'], 1.0),
    ('GOOGL', ['Google', 'Alphabet', 'Alphabet Inc', 'Google Inc', 'GOOGL', 'GOOG'], 1.0),
    ('META', ['Meta', 'Facebook', 'Meta Platforms', 'META'], 1.0),
    ('AMZN', ['Amazon', 'Amazon.com', 'AMZN'], 1.0),
    ('NVDA', ['NVIDIA', 'Nvidia Corporation', 'NVDA'], 1.0),
    ('TSLA', ['Tesla', 'Tesla Inc', 'TSLA'], 1.0),
    
    # Financial companies
    ('JPM', ['JPMorgan', 'JP Morgan', 'JPMorgan Chase', 'JPM'], 1.0),
    ('BAC', ['Bank of America', 'BofA', 'BAC'], 1.0),
    ('GS', ['Goldman Sachs', 'Goldman', 'GS'], 1.0),
    
    # Indices
    ('^GSPC', ['S&P 500', 'S&P', 'SPX', 'SP500'], 1.0),
    ('^DJI', ['Dow Jones', 'Dow', 'DJIA'], 1.0),
    ('^IXIC', ['Nasdaq', 'NASDAQ Composite', 'Nasdaq Composite'], 1.0),
    
    # Other major companies
    ('WMT', ['Walmart', 'Wal-Mart', 'WMT'], 1.0),
    ('JNJ', ['Johnson & Johnson', 'Johnson and Johnson', 'JNJ'], 1.0),
    ('PG', ['Procter & Gamble', 'P&G', 'PG'], 1.0),
    ('KO', ['Coca-Cola', 'Coke', 'KO'], 1.0),
    ('DIS', ['Disney', 'Walt Disney', 'DIS'], 1.0),
    ('NFLX', ['Netflix', 'NFLX'], 1.0),
    ('INTC', ['Intel', 'Intel Corporation', 'INTC'], 1.0),
    ('AMD', ['AMD', 'Advanced Micro Devices', 'AMD'], 1.0)
]

def init_ticker_db(db_path: str = 'data/trader_joe.db') -> None:
    """Initialize the ticker database with common stock mappings."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Drop existing table and create new one
            conn.execute('DROP TABLE IF EXISTS entities')
            conn.execute('''CREATE TABLE entities
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 ticker TEXT NOT NULL,
                 confidence REAL NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)')
            
            # Clear existing mappings
            conn.execute('DELETE FROM entities')
            
            # Insert mappings
            for ticker, names, confidence in TICKER_MAPPINGS:
                for name in names:
                    conn.execute('''INSERT INTO entities (name, ticker, confidence)
                                  VALUES (?, ?, ?)''', (name, ticker, confidence))
            
            conn.commit()
            logger.info(f"Successfully initialized ticker database at {db_path}")
            
    except Exception as e:
        logger.error(f"Failed to initialize ticker database: {e}")
        raise

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    init_ticker_db()
