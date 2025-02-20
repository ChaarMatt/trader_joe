"""
Data module for Trader Joe.
Handles database operations and data storage.
"""

import sqlite3
from pathlib import Path
from typing import Union, Optional
import os

def init_database(db_path: Union[str, Path]) -> None:
    """Initialize the SQLite database with required tables."""
    db_path = Path(db_path)
    os.makedirs(os.path.dirname(str(db_path)), exist_ok=True)
    
    # Convert Path to str for sqlite3.connect
    db_path_str = str(db_path)
    
    with sqlite3.connect(db_path_str) as conn:
        # Create articles table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                published_at TIMESTAMP NOT NULL,
                sentiment_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_published_at ON articles(published_at)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_url ON articles(url)')
        
        # Create entities table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES articles(id),
                UNIQUE(article_id, name)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)')
        
        # Create suggestions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                sentiment_score REAL NOT NULL,
                price REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_suggestion_ticker ON suggestions(ticker)')
