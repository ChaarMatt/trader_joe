import pytest
import asyncio
from src.data.market_data import MarketDataManager

@pytest.mark.asyncio
async def test_market_data_with_news():
    """Test market data analysis with news integration."""
    manager = MarketDataManager()
    
    # Test with a well-known ticker that should have news
    analysis = await manager.analyze_technicals('AAPL')
    
    # Basic assertions
    assert analysis is not None
    assert 'signal' in analysis
    assert 'confidence' in analysis
    assert 'indicators' in analysis
    
    # Check news sentiment integration
    if 'news_sentiment' in analysis:
        assert 'score' in analysis['news_sentiment']
        assert 'article_count' in analysis['news_sentiment']
        print("\nNews Sentiment Results:")
        print(f"Score: {analysis['news_sentiment']['score']:.2f}")
        print(f"Article Count: {analysis['news_sentiment']['article_count']}")
    
    # Print results for manual verification
    print("\nMarket Analysis Results:")
    print(f"Signal: {analysis['signal']}")
    print(f"Confidence: {analysis['confidence']:.2%}")
    print("Technical Indicators:")
    for indicator, value in analysis['indicators'].items():
        print(f"{indicator}: {value}")

@pytest.mark.asyncio
async def test_market_data_reliability():
    """Test market data fetching reliability."""
    manager = MarketDataManager()
    
    # Test multiple tickers to verify consistent behavior
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in tickers:
        # Get market data
        try:
            # Add delay between requests to respect rate limits
            await asyncio.sleep(5)  # 5 second delay between tickers
            data = await manager.get_market_data(ticker)
            assert data is not None
            assert not data.empty
            
            # Verify technical indicators are calculated
            assert 'SMA_20' in data.columns
            assert 'RSI_14' in data.columns
            assert 'MACD' in data.columns
            assert 'Signal_Line' in data.columns
        except Exception as e:
            pytest.fail(f"Failed to get market data for {ticker}: {e}")
            
        # Get technical analysis
        analysis = await manager.analyze_technicals(ticker)
        assert analysis is not None
        assert analysis['signal'] in ['BUY', 'SELL', 'NEUTRAL']
        assert 0 <= analysis['confidence'] <= 1
        
        print(f"\nReliability Test Results for {ticker}:")
        print(f"Signal: {analysis['signal']}")
        print(f"Confidence: {analysis['confidence']:.2%}")
        if 'news_sentiment' in analysis:
            print(f"News Articles Found: {analysis['news_sentiment']['article_count']}")

if __name__ == '__main__':
    asyncio.run(test_market_data_with_news())
    asyncio.run(test_market_data_reliability())
