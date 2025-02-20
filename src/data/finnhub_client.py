"""
Finnhub API client for real-time market data with advanced rate limiting and caching.
"""

import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, TypeVar, cast
from datetime import datetime, timedelta
import yaml
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

T = TypeVar('T')

class FinnhubClient:
    """Client for interacting with Finnhub API with advanced rate limiting."""
    
    def __init__(self, api_key: str):
        """Initialize the Finnhub client."""
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.headers = {
            'X-Finnhub-Token': api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'TraderJoe/1.0'
        }
        
        # Initialize rate limiter (30 requests per second per endpoint)
        self._rate_limiter = RateLimiter()
        for endpoint in ['quote', 'stock/profile2', 'stock/price-target', 'stock/recommendation']:
            self._rate_limiter.add_limit(endpoint, 30, 1.0)
        
        # Cache durations
        self._cache_durations = {
            'quote': timedelta(seconds=1),
            'stock/profile2': timedelta(days=1),
            'stock/price-target': timedelta(hours=1),
            'stock/recommendation': timedelta(hours=1)
        }
        
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Make a request to the Finnhub API with rate limiting and retries."""
        max_retries = 3
        base_delay = 1.0  # Base delay for exponential backoff
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                wait_time = await self._rate_limiter.acquire(endpoint)
                if wait_time > 0:
                    logger.debug(f"Rate limit hit, waiting {wait_time:.2f}s before retry")
                    await asyncio.sleep(wait_time)
                
                # Check cache first
                cache_key = f"{endpoint}:{str(params)}"
                cached = self._rate_limiter.cache_get(cache_key, self._cache_durations[endpoint])
                if cached is not None:
                    return cached
                
                url = f"{self.base_url}/{endpoint}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers, params=params) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                            logger.warning(f"Rate limit exceeded, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                            
                        if response.status != 200:
                            logger.error(f"Finnhub API error: Status {response.status}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(base_delay * (2 ** attempt))
                                continue
                            return None
                            
                        data = await response.json()
                        # Cache successful response
                        self._rate_limiter.cache_set(cache_key, data, self._cache_durations[endpoint])
                        return data
                        
            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
            
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol."""
        try:
            response = await self._make_request('quote', {'symbol': symbol})
            return cast(Dict[str, Any], response) if isinstance(response, dict) else None
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None
            
    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile information."""
        try:
            response = await self._make_request('stock/profile2', {'symbol': symbol})
            return cast(Dict[str, Any], response) if isinstance(response, dict) else None
        except Exception as e:
            logger.error(f"Failed to get company profile for {symbol}: {e}")
            return None
            
    async def get_price_target(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price target data."""
        try:
            response = await self._make_request('stock/price-target', {'symbol': symbol})
            return cast(Dict[str, Any], response) if isinstance(response, dict) else None
        except Exception as e:
            logger.error(f"Failed to get price target for {symbol}: {e}")
            return None
            
    async def get_recommendation_trends(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get analyst recommendation trends."""
        try:
            response = await self._make_request('stock/recommendation', {'symbol': symbol})
            return cast(List[Dict[str, Any]], response) if isinstance(response, list) else None
        except Exception as e:
            logger.error(f"Failed to get recommendations for {symbol}: {e}")
            return None

if __name__ == '__main__':
    async def main():
        # Load config to get API key
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        api_key = config.get('api_keys', {}).get('finnhub')
        if not api_key:
            print("Finnhub API key not found in config")
            return
            
        client = FinnhubClient(api_key)
        
        # Example usage
        symbol = 'AAPL'
        quote = await client.get_quote(symbol)
        profile = await client.get_company_profile(symbol)
        price_target = await client.get_price_target(symbol)
        recommendations = await client.get_recommendation_trends(symbol)
        
        print(f"\nQuote for {symbol}:")
        print(quote)
        print(f"\nCompany Profile:")
        print(profile)
        print(f"\nPrice Target:")
        print(price_target)
        print(f"\nRecommendation Trends:")
        print(recommendations)
    
    asyncio.run(main())
