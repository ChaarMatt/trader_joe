import asyncio
import pytest
from src.data.ticker_mapper import TickerMapper
import os
import sqlite3
import json

@pytest.fixture
def ticker_mapper():
    db_path = "data/trader_joe.db"
    mapper = TickerMapper(db_path)
    # Pre-populate cache with test data
    with sqlite3.connect(mapper.db_path) as conn:
        # Create tables if they don't exist
        mapper._setup_database()
        
        # Insert test data
        test_info = {
            "longName": "Apple Inc.",
            "regularMarketPrice": 150.0,
            "marketCap": 2000000000000
        }
        conn.execute(
            '''INSERT OR REPLACE INTO verified_tickers 
               (ticker, info, last_verified)
               VALUES (?, ?, datetime('now'))''',
            ('AAPL', json.dumps(test_info))
        )
        conn.execute(
            '''INSERT OR REPLACE INTO ticker_mappings
               (company_name, ticker, confidence, last_verified)
               VALUES (?, ?, ?, datetime('now'))''',
            ('Apple', 'AAPL', 1.0)
        )
        conn.commit()
    return mapper

def test_extract_companies_from_text(ticker_mapper):
    text = """
    Apple Inc. announced a new partnership with Microsoft Corporation today. 
    The deal will help both tech giants compete with Amazon.com in the cloud space.
    $AAPL stock rose 2% while MSFT (Microsoft) gained 1.5%.
    """
    companies = ticker_mapper._extract_companies_from_text(text)
    assert "Apple Inc." in companies
    assert "Microsoft Corporation" in companies
    assert "Amazon.com" in companies

def test_extract_potential_tickers(ticker_mapper):
    text = """
    $AAPL stock is trading at $150 today.
    MSFT (Microsoft) reported earnings.
    NYSE:GOOGL fell 2%.
    """
    tickers = ticker_mapper._extract_potential_tickers(text)
    assert ("AAPL", 1.0) in tickers  # $AAPL format
    assert ("MSFT", 0.9) in tickers  # Company (Ticker) format
    assert ("GOOGL", 1.0) in tickers  # Exchange:Ticker format

def test_normalize_company_name(ticker_mapper):
    variants = [
        "Apple Inc.",
        "Apple Corporation",
        "Apple, Inc",
        "APPLE INC",
    ]
    normalized = [ticker_mapper._normalize_company_name(v) for v in variants]
    # All variants should normalize to the same name
    assert len(set(normalized)) == 1
    assert "Apple" in normalized[0]

@pytest.mark.asyncio
async def test_get_ticker(ticker_mapper):
    # Test with direct company name and pre-populated cache
    result = await ticker_mapper.get_ticker("Apple Inc")
    assert result is not None
    ticker, confidence = result
    assert ticker == "AAPL"
    assert confidence > 0.8

    # Test with context
    result = await ticker_mapper.get_ticker(
        "Apple",
        "Apple Inc. ($AAPL) announced strong earnings today. The tech giant's stock..."
    )
    assert result is not None
    ticker, confidence = result
    assert ticker == "AAPL"
    assert confidence > 0.9

if __name__ == "__main__":
    pytest.main([__file__])
