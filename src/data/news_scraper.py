"""
News scraping functionality for Trader Joe.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import aiohttp
from bs4 import BeautifulSoup, Tag
import yaml
from src.utils.text_processing import clean_text
from src.analysis.sentiment_analysis import SentimentAnalyzer
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class NewsAPIClient:
    """Client for NewsAPI integration with advanced rate limiting."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.headers = {
            'X-Api-Key': api_key,
            'User-Agent': 'TraderJoe/1.0'
        }
        
        # Initialize rate limiter (500 requests per day = ~0.006 requests per second)
        # Allow more frequent requests but with longer cooldown periods
        self._rate_limiter = RateLimiter()
        self._rate_limiter.add_limit('everything', 1, 0.006)  # Increased rate
        
        # Cache duration for news (30 minutes to reduce API calls)
        self._cache_duration = timedelta(minutes=30)

    async def get_financial_news(self) -> List[Dict[str, Any]]:
        """Fetch financial news from NewsAPI with caching and rate limiting."""
        try:
            # Define queries
            queries = [
                'stock market',
                'financial markets',
                'trading',
                'stocks',
                'finance'
            ]
            
            # Create cache key based on queries and time window
            from_date = (datetime.now() - timedelta(days=1)).isoformat()
            cache_key = f"news_api:{','.join(queries)}:{from_date}"
            
            # Check cache
            cached = self._rate_limiter.cache_get(cache_key, self._cache_duration)
            if cached is not None:
                logger.info("Returning cached news articles")
                return cached
            
            # Prepare request
            url = f"{self.base_url}/everything"
            combined_query = ' OR '.join(f'"{q}"' for q in queries)
            params = {
                'q': combined_query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'from': from_date
            }
            
            articles = []
            retry_count = 0
            
            while await self._rate_limiter.wait_and_retry('everything', retry_count):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=self.headers, params=params) as response:
                            if response.status == 429:  # Too Many Requests
                                # Get retry-after from headers or use default
                                retry_after = int(response.headers.get('Retry-After', 60))
                                logger.warning(f"Rate limit exceeded, waiting {retry_after}s before retry {retry_count + 1}")

                                await asyncio.sleep(retry_after)
                                retry_count += 1
                                continue
                                
                            if response.status != 200:
                                logger.error(f"NewsAPI error: Status {response.status}")
                                if retry_count < 2:  # Allow a couple retries for non-429 errors
                                    retry_count += 1
                                    await asyncio.sleep(5)  # Short delay before retry
                                    continue
                                return []
                                
                            data = await response.json()
                            if data.get('status') != 'ok':
                                logger.error(f"API error: {data.get('message')}")
                                return []
                            
                            if 'articles' in data:
                                raw_articles = data['articles']
                                # Process articles
                                for article in raw_articles:
                                    if article.get('title') and article.get('description'):
                                        articles.append({
                                            'title': article['title'],
                                            'content': article['description'],
                                            'url': article.get('url', ''),
                                            'source': article.get('source', {}).get('name', 'NewsAPI'),
                                            'published_at': article.get('publishedAt', datetime.now().isoformat())
                                        })
                                
                                # Update rate limit based on response headers
                                if 'X-RateLimit-Remaining' in response.headers:
                                    remaining = int(response.headers['X-RateLimit-Remaining'])
                                    if remaining < 10:  # If running low on requests
                                        logger.warning(f"Running low on NewsAPI requests: {remaining} remaining")
                                        # Increase cooldown period
                                        self._rate_limiter.add_limit('everything', 1, 0.003)  # More conservative rate
                                
                                # Cache successful response
                                self._rate_limiter.cache_set(cache_key, articles, self._cache_duration)
                                logger.info(f"Successfully processed {len(articles)} articles from NewsAPI")
                                return articles
                                
                            logger.warning("No articles found in NewsAPI response")
                            return []
                            
                except aiohttp.ClientError as e:
                    logger.error(f"Request failed: {e}")
                    retry_count += 1
                    if "Too Many Requests" not in str(e):
                        return []
                    continue
                    
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    return []
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to fetch news from NewsAPI: {e}")
            return []

class NewsScraper:
    """Scrapes news articles from various sources."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = self._load_config(config_path)
        # Get sources from config
        self.sources = []
        for source in self.config.get('news_sources', []):
            if source['name'] == 'Yahoo Finance':
                self.sources.extend([
                    'https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US',
                    'https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US',
                    'https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US'
                ])
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        self.newsapi_client = None
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize NewsAPI client if key is available
        newsapi_key = self.config.get('api_keys', {}).get('newsapi')
        if newsapi_key:
            logger.info("Initializing NewsAPI client")
            self.newsapi_client = NewsAPIClient(newsapi_key)
        else:
            logger.warning("NewsAPI key not found in config, NewsAPI integration will be disabled")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    async def scrape_news(self) -> List[Dict[str, Any]]:
        """Scrape news articles from all configured sources and analyze sentiment."""
        try:
            articles = []
            
            # Get news from Yahoo Finance
            # Scrape Yahoo Finance RSS feeds
            timeout = aiohttp.ClientTimeout(total=30)
            conn = aiohttp.TCPConnector(force_close=True, limit=100, ssl=False)
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # Scrape Yahoo Finance
            async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=headers) as session:
                for source in self.sources:
                    try:
                        async with session.get(source) as response:
                            if response.status != 200:
                                logger.warning(f"Failed to fetch {source}: Status {response.status}")
                                continue
                            
                            xml = await response.text()
                            soup = BeautifulSoup(xml, 'lxml-xml')
                            items = soup.find_all('item')
                            
                            if not items:
                                logger.warning(f"No items found in feed: {source}")
                                continue
                                
                            for item in items:
                                try:
                                    if not isinstance(item, Tag):
                                        continue
                                        
                                    title_elem = item.find('title')
                                    desc_elem = item.find('description')
                                    link_elem = item.find('link')
                                    date_elem = item.find('pubDate')
                                    
                                    title = title_elem.get_text().strip() if isinstance(title_elem, Tag) else None
                                    description = desc_elem.get_text().strip() if isinstance(desc_elem, Tag) else None
                                    url = link_elem.get_text().strip() if isinstance(link_elem, Tag) else None
                                    pub_date = date_elem.get_text().strip() if isinstance(date_elem, Tag) else None
                                    
                                    if not all([title, description, url]):
                                        continue
                                    
                                    content = clean_text(str(description))
                                    published_at = str(pub_date) if pub_date else datetime.now().isoformat()
                                    
                                    # Analyze sentiment
                                    sentiment_result = await self.sentiment_analyzer.analyze_sentiment(f"{title} {content}")
                                    
                                    articles.append({
                                        'title': title,
                                        'content': content,
                                        'url': url,
                                        'source': 'Yahoo Finance',
                                        'published_at': published_at,
                                        'sentiment_score': sentiment_result['sentiment_score'],
                                        'sentiment': sentiment_result['sentiment']
                                    })
                                    
                                    await asyncio.sleep(0.1)
                                except Exception as e:
                                    logger.error(f"Failed to parse article: {e}")
                                    continue
                    except Exception as e:
                        logger.error(f"Failed to scrape {source}: {e}")
                        continue
            
            # Get news from NewsAPI if available
            if self.newsapi_client:
                try:
                    logger.info("Fetching news from NewsAPI...")
                    newsapi_articles = await self.newsapi_client.get_financial_news()
                    logger.info(f"Fetched {len(newsapi_articles)} articles from NewsAPI")
                    
                    for article in newsapi_articles:
                        try:
                            # Analyze sentiment
                            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(
                                f"{article['title']} {article['content']}"
                            )
                            article['sentiment_score'] = sentiment_result['sentiment_score']
                            article['sentiment'] = sentiment_result['sentiment']
                            articles.append(article)
                        except Exception as e:
                            logger.error(f"Failed to process NewsAPI article: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Failed to get NewsAPI articles: {e}")
            else:
                logger.info("NewsAPI client not initialized, skipping NewsAPI news fetch")
            
            return articles
        except Exception as e:
            logger.error(f"Failed to scrape news: {e}")
            return []

if __name__ == '__main__':
    # Example usage
    async def main():
        scraper = NewsScraper()
        articles = await scraper.scrape_news()
        for article in articles:
            print(f"Title: {article['title']}\nContent: {article['content']}\n")
    
    asyncio.run(main())
