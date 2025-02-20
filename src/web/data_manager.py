"""
Data manager for efficient data fetching and caching.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from src.data.market_data import MarketDataManager
from src.data.news_scraper import NewsScraper
from src.engine.recommendation import RecommendationEngine

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data fetching and caching for the dashboard."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the data manager."""
        self.market_data = MarketDataManager(config_path)
        self.news_scraper = NewsScraper(config_path)
        self.recommendation_engine = RecommendationEngine(config_path)
        
        # Cache storage
        self._cache: Dict[str, Dict[str, Any]] = {
            'market_data': {},
            'news': {},
            'suggestions': {}
        }
        
        # Cache durations - longer cache times to reduce API calls
        self._cache_durations = {
            'market_data': timedelta(minutes=1),  # Cache market data for 1 minute
            'news': timedelta(minutes=10),        # Cache news for 10 minutes
            'suggestions': timedelta(minutes=5)    # Cache suggestions for 5 minutes
        }
        
        # Batch processing - smaller batches with longer delays
        self._batch_size = 3  # Process 3 symbols at a time
        self._batch_delay = 2.0  # 2 second delay between batches
        
    def _is_cache_valid(self, cache_type: str, key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_type not in self._cache or key not in self._cache[cache_type]:
            return False
            
        cache_entry = self._cache[cache_type][key]
        if 'timestamp' not in cache_entry:
            return False
            
        age = datetime.now() - cache_entry['timestamp']
        return age < self._cache_durations[cache_type]
        
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market data for multiple symbols with batching and caching."""
        result = {}
        symbols_to_fetch = []
        
        # Check cache first
        for symbol in symbols:
            if self._is_cache_valid('market_data', symbol):
                result[symbol] = self._cache['market_data'][symbol]['data']
            else:
                symbols_to_fetch.append(symbol)
                
        # Fetch new data in batches with sequential processing
        for i in range(0, len(symbols_to_fetch), self._batch_size):
            batch = symbols_to_fetch[i:i + self._batch_size]
            
            # Process each symbol sequentially to avoid overwhelming the API
            for symbol in batch:
                try:
                    data = await self.market_data.get_market_data(symbol)
                    if data is not None:
                        self._cache['market_data'][symbol] = {
                            'data': data,
                            'timestamp': datetime.now()
                        }
                        result[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
            
            # Add delay between batches
            if i + self._batch_size < len(symbols_to_fetch):
                await asyncio.sleep(self._batch_delay)
                
        return result
        
    async def get_news(self) -> List[Dict[str, Any]]:
        """Get news with caching."""
        cache_key = 'latest'
        if self._is_cache_valid('news', cache_key):
            return self._cache['news'][cache_key]['data']
            
        try:
            news = await self.news_scraper.scrape_news()
            if news:
                self._cache['news'][cache_key] = {
                    'data': news,
                    'timestamp': datetime.now()
                }
                return news
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            
        return []
        
    async def get_trading_suggestions(self) -> List[Dict[str, Any]]:
        """Get trading suggestions with caching."""
        cache_key = 'latest'
        if self._is_cache_valid('suggestions', cache_key):
            return self._cache['suggestions'][cache_key]['data']
            
        try:
            # Get news for suggestions
            news = await self.get_news()
            suggestions = await self.recommendation_engine.generate_suggestions(news)
            
            if suggestions:
                self._cache['suggestions'][cache_key] = {
                    'data': [s.to_dict() for s in suggestions],
                    'timestamp': datetime.now()
                }
                return self._cache['suggestions'][cache_key]['data']
                
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            
        return []
        
    def invalidate_cache(self, cache_type: Optional[str] = None):
        """Invalidate cache entries."""
        if cache_type:
            if cache_type in self._cache:
                self._cache[cache_type].clear()
        else:
            for cache in self._cache.values():
                cache.clear()
