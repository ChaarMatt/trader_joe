"""
Recommendation engine for Trader Joe.
Generates trading suggestions based on sentiment analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3
import yaml
import asyncio
from datetime import datetime, timedelta
from src.data.ticker_mapper import TickerMapper
from src.data.market_data import MarketDataManager
from src.analysis.deep_optimizer import DeepTradeOptimizer

logger = logging.getLogger(__name__)

class TradingSuggestion:
    """Data class for trading suggestions."""
    
    def __init__(
        self,
        ticker: str,
        action: str,
        confidence: float,
        sentiment_score: float,
        price: float,
        supporting_articles: List[Dict[str, Any]]
    ):
        self.ticker = ticker
        self.action = action
        self.confidence = confidence
        self.sentiment_score = sentiment_score
        self.price = price
        self.supporting_articles = supporting_articles
        self.timestamp = datetime.now()
        self.optimization_notes = {}  # Store AI optimization details

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to dictionary format."""
        return {
            'ticker': self.ticker,
            'action': self.action,
            'confidence': self.confidence,
            'sentiment_score': self.sentiment_score,
            'price': self.price,
            'supporting_articles': self.supporting_articles,
            'timestamp': self.timestamp.isoformat(),
            'optimization_notes': self.optimization_notes
        }

class NewsSentimentRecommender:
    """Generates trading suggestions based on news sentiment analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        self._config: Dict[str, Any] = self._load_config(config_path)
        self._sentiment_threshold: float = self._config.get('trading', {}).get('sentiment_threshold', 0.5)
        self._max_recommendations: int = self._config.get('trading', {}).get('max_recommendations', 5)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            'trading': {
                'sentiment_threshold': 0.5,
                'max_recommendations': 5
            },
            'database': {
                'path': 'data/trader_joe.db'
            }
        }
        
        if not config_path:
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return default_config

    async def generate_suggestions(self, news_sentiment_data: List[Dict[str, Any]]) -> List[TradingSuggestion]:
        """Generate trading suggestions based on news sentiment and technical analysis."""
        if not news_sentiment_data:
            return []

        ticker_mapper = TickerMapper(self._config.get('database', {}).get('path', 'data/trader_joe.db'))
        market_data = MarketDataManager()
        
        # Group articles by ticker
        ticker_articles: Dict[str, List[Dict[str, Any]]] = {}
        for article in news_sentiment_data:
            if abs(article['sentiment_score']) < self._sentiment_threshold:
                continue
                
            result = await ticker_mapper.get_ticker(article['content'])
            if result:
                ticker, _ = result
                if ticker not in ticker_articles:
                    ticker_articles[ticker] = []
                ticker_articles[ticker].append(article)
        
        # Generate suggestions
        suggestions = []
        for ticker, articles in ticker_articles.items():
            try:
                # Calculate average sentiment
                sentiment_scores = [article['sentiment_score'] for article in articles]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiment_confidence = min(abs(avg_sentiment), 1.0)
                
                # Get technical analysis
                tech_analysis = await market_data.analyze_technicals(ticker)
                
                if tech_analysis['signal'] == 'NEUTRAL':
                    continue
                    
                # Combine signals
                sentiment_signal = 1 if avg_sentiment > 0 else -1
                tech_signal = 1 if tech_analysis['signal'] == 'BUY' else -1
                
                # Only generate suggestion if signals agree
                if sentiment_signal == tech_signal:
                    action = 'BUY' if sentiment_signal > 0 else 'SELL'
                    
                    # Combine confidences (40% sentiment, 60% technical)
                    combined_confidence = (
                        0.4 * sentiment_confidence +
                        0.6 * tech_analysis['confidence']
                    )
                    
                    suggestion = TradingSuggestion(
                        ticker=ticker,
                        action=action,
                        confidence=combined_confidence,
                        sentiment_score=avg_sentiment,
                        price=tech_analysis['indicators']['price'],
                        supporting_articles=articles[:3]  # Include top 3 articles
                    )
                    suggestions.append(suggestion)
                    
                    if len(suggestions) >= self._max_recommendations:
                        break
                        
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
                continue
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions

class RecommendationEngine:
    """Generates trading suggestions based on sentiment analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        self._config: Dict[str, Any] = self._load_config(config_path)
        
        # Initialize components
        self._news_recommender = NewsSentimentRecommender(config_path)
        
        self._market_data = MarketDataManager()
            
        # Initialize deep optimizer with market data instance
        optimizer_config = self._config.copy()
        optimizer_config['market_data'] = self._market_data
        self._deep_optimizer = DeepTradeOptimizer(optimizer_config)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            'trading': {
                'sentiment_threshold': 0.5,
                'max_recommendations': 5
            },
            'database': {
                'path': 'data/trader_joe.db'
            }
        }
        
        if not config_path:
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return default_config

    async def generate_suggestions(self, news_data: List[Dict[str, Any]]) -> List[TradingSuggestion]:
        """Generate trading suggestions based on news data and optimize with deep learning."""
        if not self._news_recommender:
            raise ValueError("News recommender not initialized")

        try:
            # Get initial suggestions from news sentiment analysis
            initial_suggestions = await self._news_recommender.generate_suggestions(news_data)
            
            # Optimize each suggestion using deep learning
            optimized_suggestions = []
            for suggestion in initial_suggestions:
                # Get optimized data
                optimized_data = await self._deep_optimizer.optimize_trading_decision(
                    suggestion.ticker,
                    suggestion.supporting_articles,
                    suggestion.to_dict()
                )
                
                # Create new suggestion with optimized data
                optimized_suggestion = TradingSuggestion(
                    ticker=optimized_data['ticker'],
                    action=optimized_data['action'],
                    confidence=optimized_data['confidence'],
                    sentiment_score=optimized_data['sentiment_score'],
                    price=optimized_data['price'],
                    supporting_articles=optimized_data['supporting_articles']
                )
                
                # Add AI optimization notes
                setattr(optimized_suggestion, 'optimization_notes', optimized_data.get('optimization_notes', {}))
                optimized_suggestions.append(optimized_suggestion)
            
            return optimized_suggestions
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        # Initialize recommendation engine
        engine = RecommendationEngine()
        
        # Example news data
        news_data = [
            {
                'title': 'Positive Earnings Report',
                'content': 'Company X reported strong earnings growth.',
                'sentiment_score': 0.8,
                'current_price': 100.0
            }
        ]
        
        # Generate suggestions
        suggestions = await engine.generate_suggestions(news_data)
        
        # Print suggestions
        for suggestion in suggestions:
            print(f"{suggestion.ticker}: {suggestion.action} (confidence: {suggestion.confidence:.2%})")

    # Run the async main function
    asyncio.run(main())
