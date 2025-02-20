"""
Main application entry point for Trader Joe.
Integrates news analysis, market data, and trading recommendations.
"""

import logging
import yaml
from typing import List, Dict, Any
import asyncio
from datetime import datetime
from threading import Thread
from src.engine.recommendation import RecommendationEngine
from src.data.news_scraper import NewsScraper
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.data.market_data import MarketDataManager
from src.web.app import init_app

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config if config is not None else {}
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}

def run_web_app(config: Dict[str, Any]):
    """Run the web application."""
    server, socketio, _ = init_app(config)
    socketio.run(
        server,
        host=config['web']['host'],
        port=config['web']['port'],
        debug=config['web']['debug']
    )

async def analyze_market(market_data: MarketDataManager, ticker: str) -> Dict[str, Any]:
    """Analyze market data with retry logic for rate limits."""
    default_result = {
        'signal': 'NEUTRAL',
        'confidence': 0.0,
        'indicators': {}
    }
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            result = await market_data.analyze_technicals(ticker)
            return result if result is not None else default_result
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Rate limit hit for {ticker}, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed to analyze {ticker}: {e}")
                if attempt == max_retries - 1:
                    return default_result
                
    return default_result

async def main():
    """Main application entry point."""
    try:
        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration")
            return

        # Start web application in a separate thread
        web_thread = Thread(target=run_web_app, args=(config,), daemon=True)
        web_thread.start()
        logger.info("Web dashboard started")

        # Initialize components with config
        news_scraper = NewsScraper(config_path='config/config.yaml')
        sentiment_analyzer = SentimentAnalyzer()
        recommendation_engine = RecommendationEngine(config_path='config/config.yaml')
        market_data = MarketDataManager(config_path='config/config.yaml')

        while True:
            try:
                # Scrape news from all sources
                logger.info("Fetching news data...")
                news_data = await news_scraper.scrape_news()
                logger.info(f"Fetched {len(news_data)} news articles")

                # Extract content for sentiment analysis
                news_contents = [f"{article['title']} {article['content']}" for article in news_data]
                
                # Analyze sentiment in batch
                logger.info("Analyzing sentiment...")
                sentiment_results = await sentiment_analyzer.analyze_sentiment_batch(news_contents)
                
                # Combine news data with sentiment results
                news_with_sentiment = []
                for article, sentiment in zip(news_data, sentiment_results):
                    article_with_sentiment = article.copy()
                    article_with_sentiment.update(sentiment)
                    news_with_sentiment.append(article_with_sentiment)

                # Generate recommendations
                logger.info("Generating trading recommendations...")
                recommendations = await recommendation_engine.generate_suggestions(news_with_sentiment)

                # Print recommendations with detailed market data
                print("\nTrading Recommendations:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("-" * 80)
                
                for recommendation in recommendations:
                    # Get real-time market data with retry logic
                    market_analysis = await analyze_market(market_data, recommendation.ticker)
                    
                    print(f"\nTicker: {recommendation.ticker}")
                    print(f"Action: {recommendation.action}")
                    print(f"Confidence: {recommendation.confidence:.2%}")
                    print(f"Sentiment Score: {recommendation.sentiment_score:.2f}")
                    
                    # Print market data if available
                    if market_analysis and market_analysis.get('indicators'):
                        indicators = market_analysis['indicators']
                        print("\nMarket Data:")
                        print(f"Current Price: ${indicators['price']:.2f}")
                        print(f"RSI (14): {indicators['rsi_14']:.2f}")
                        print(f"MACD: {indicators['macd']:.4f}")
                        
                        if 'real_time' in market_analysis:
                            rt = market_analysis['real_time']
                            print(f"Real-time Change: {rt['percent_change']}%")
                            
                    # Print supporting articles
                    if recommendation.supporting_articles:
                        print("\nSupporting Articles:")
                        for article in recommendation.supporting_articles:
                            print(f"- {article['title']} (Sentiment: {article.get('sentiment', 'N/A')})")
                    
                    print("-" * 80)

                # Wait before next update
                update_interval = config.get('trading', {}).get('update_interval_seconds', 300)
                logger.info(f"Waiting {update_interval} seconds before next update...")
                await asyncio.sleep(update_interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    except Exception as e:
        logger.error(f"Error in main application: {e}")
        raise

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    asyncio.run(main())
