"""
Market data functionality for Trader Joe.
Handles fetching and analyzing market data.
"""

import logging
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.data.finnhub_client import FinnhubClient
from src.data.news_scraper import NewsScraper
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class MarketDataManager:
    """Manages market data fetching and technical analysis."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the market data manager."""
        self._config = self._load_config(config_path)
        self.news_scraper = NewsScraper(config_path)
        
        # Initialize rate limiter
        self._rate_limiter = RateLimiter()
        # YFinance: Allow larger bursts with reasonable refill
        self._rate_limiter.add_limit('yfinance', 30, 0.5)  # 30 tokens, 0.5 per second = more responsive
        # Special rate limit for indices - more frequent updates
        self._rate_limiter.add_limit('index', 5, 0.2)  # 5 requests per 5 seconds for indices
        # Separate Finnhub rate limits
        self._rate_limiter.add_limit('finnhub_quote', 30, 1.0)  # 30 requests per second for quotes
        self._rate_limiter.add_limit('finnhub_other', 10, 0.5)  # 10 requests per 2 seconds for other endpoints
        
        # Cache durations
        self._cache_durations = {
            'market_data': timedelta(minutes=1),  # More frequent updates for real-time data
            'technical_analysis': timedelta(minutes=5),
            'historical_data': timedelta(minutes=30),
            'index_data': timedelta(minutes=5)  # More frequent index updates
        }
        
        # Initialize Finnhub client if key is available
        self.finnhub_client = None
        finnhub_key = self._config.get('api_keys', {}).get('finnhub')
        if finnhub_key:
            self.finnhub_client = FinnhubClient(finnhub_key)
            
    async def _get_data_with_rate_limit(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Get data with advanced rate limiting and retries using enhanced rate limiter."""
        # Check cache first
        cache_key = f"yfinance:{ticker}:{period}:{interval}"
        cached = self._rate_limiter.cache_get(cache_key, self._cache_durations['market_data'])
        if cached is not None:
            return cached

        # Use different rate limits for indices vs stocks
        endpoint = 'index' if ticker.startswith('^') else 'yfinance'
        cache_duration = self._cache_durations['index_data'] if ticker.startswith('^') else self._cache_durations['market_data']
        
        retry_count = 0
        while await self._rate_limiter.wait_and_retry(endpoint, retry_count):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period=period, interval=interval)
                
                if df.empty:
                    raise Exception("Empty DataFrame returned")
                
                # Cache successful response
                self._rate_limiter.cache_set(cache_key, df, cache_duration)
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {retry_count + 1} failed for {ticker}: {e}")
                retry_count += 1
                
                if "Too Many Requests" not in str(e) or retry_count >= 3:
                    logger.error(f"Failed to get data for {ticker}: {e}")
                    return None
                    
        return None
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
        
    async def get_real_time_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote data from Finnhub with improved rate limiting."""
        if not self.finnhub_client:
            return None
            
        retry_count = 0
        while await self._rate_limiter.wait_and_retry('finnhub_quote', retry_count):
            try:
                quote = await self.finnhub_client.get_quote(ticker)
                if quote:
                    return {
                        'current_price': quote.get('c'),
                        'change': quote.get('d'),
                        'percent_change': quote.get('dp'),
                        'high': quote.get('h'),
                        'low': quote.get('l'),
                        'open': quote.get('o'),
                        'previous_close': quote.get('pc'),
                        'timestamp': quote.get('t')
                    }
                return None
            except Exception as e:
                logger.error(f"Failed to get real-time quote for {ticker}: {e}")
                retry_count += 1
                if retry_count >= 3:
                    return None
                continue
        
        return None

    async def get_market_data(self, ticker: str, period: str = '1mo', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get market data with improved caching and rate limiting."""
        try:
            cache_key = f"market_data:{ticker}:{period}:{interval}"
            cached = self._rate_limiter.cache_get(cache_key, self._cache_durations['market_data'])
            if cached is not None:
                return cached
            
            # Try to get real-time data from Finnhub first
            if self.finnhub_client:
                try:
                    quote = await self.get_real_time_quote(ticker)
                    if quote:
                        df = await self._get_data_with_rate_limit(ticker, period, interval)
                        if df is not None and not df.empty:
                            # Update the latest price with real-time data
                            df.loc[df.index[-1], 'Close'] = quote['current_price']
                            df.loc[df.index[-1], 'High'] = max(df.loc[df.index[-1], 'High'], quote['current_price'])
                            df.loc[df.index[-1], 'Low'] = min(df.loc[df.index[-1], 'Low'], quote['current_price'])
                            
                            # Cache the updated data
                            self._rate_limiter.cache_set(cache_key, df, self._cache_durations['market_data'])
                            return df
                except Exception as e:
                    logger.error(f"Failed to get Finnhub data for {ticker}: {e}")
            
            # Fallback to yfinance
            df = await self._get_data_with_rate_limit(ticker, period, interval)
            if df is not None and not df.empty:
                self._rate_limiter.cache_set(cache_key, df, self._cache_durations['market_data'])
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get market data for {ticker}: {e}")
            return None
            
    async def _get_historical_data(self, ticker: str, period: str = '1mo', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get historical market data from yfinance with improved caching and rate limiting."""
        # Check cache
        cache_key = f"historical:{ticker}:{period}:{interval}"
        cached = self._rate_limiter.cache_get(cache_key, self._cache_durations['historical_data'])
        if cached is not None:
            return cached

        retry_count = 0
        while await self._rate_limiter.wait_and_retry('finnhub_other', retry_count):
            try:
                # Fetch data from Yahoo Finance
                stock = yf.Ticker(ticker)
                df = stock.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data found for {ticker}")
                    return None
                    
                # Calculate technical indicators
                df = self._calculate_indicators(df)
                
                # Cache the data
                self._rate_limiter.cache_set(cache_key, df, self._cache_durations['historical_data'])
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {retry_count + 1} failed for {ticker}: {e}")
                retry_count += 1
                
                if "Too Many Requests" not in str(e) or retry_count >= 3:
                    logger.error(f"Failed to get historical data for {ticker}: {e}")
                    return None
                    
        return None
            
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            # Calculate SMA-20
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            # Calculate RSI-14
            # Convert to numpy array for calculations
            close_prices = df['Close'].to_numpy()
            delta = np.diff(close_prices, prepend=close_prices[0])
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            # Convert back to pandas series for rolling calculations
            gain = pd.Series(gains).rolling(window=14).mean()
            loss = pd.Series(losses).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return df
            
    async def analyze_technicals(self, ticker: str) -> Dict[str, Any]:
        """Analyze technical indicators and return signals. Incorporates Finnhub analyst recommendations and news sentiment."""
        try:
            df = await self.get_market_data(ticker)
            if df is None or df.empty:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'indicators': {}
                }
                
            # Get latest values
            current_price = df['Close'].iloc[-1]
            sma_20 = df['SMA_20'].iloc[-1]
            rsi_14 = df['RSI_14'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            signal_line = df['Signal_Line'].iloc[-1]
            
            # Analyze signals
            signals = []
            confidences = []
            
            # Technical Analysis Signals
            # SMA Signal
            if current_price > sma_20:
                signals.append(1)  # Bullish
                confidences.append(min((current_price / sma_20 - 1) * 5, 1.0))
            else:
                signals.append(-1)  # Bearish
                confidences.append(min((sma_20 / current_price - 1) * 5, 1.0))
                
            # RSI Signal
            if rsi_14 > 70:
                signals.append(-1)  # Overbought
                confidences.append(min((rsi_14 - 70) / 30, 1.0))
            elif rsi_14 < 30:
                signals.append(1)  # Oversold
                confidences.append(min((30 - rsi_14) / 30, 1.0))
            else:
                signals.append(0)
                confidences.append(0.3)
                
            # MACD Signal
            if macd > signal_line:
                signals.append(1)  # Bullish
                confidences.append(min((macd - signal_line) / signal_line * 10, 1.0))
            else:
                signals.append(-1)  # Bearish
                confidences.append(min((signal_line - macd) / signal_line * 10, 1.0))
                
            # Get Finnhub analyst recommendations if available
            if self.finnhub_client:
                try:
                    recommendations = await self.finnhub_client.get_recommendation_trends(ticker)
                    if isinstance(recommendations, list) and len(recommendations) > 0:
                        latest_rec = recommendations[0]
                        # Calculate analyst sentiment (-1 to 1)
                        total_ratings = sum([
                            latest_rec.get('strongBuy', 0),
                            latest_rec.get('buy', 0),
                            latest_rec.get('hold', 0),
                            latest_rec.get('hold', 0),
                            latest_rec.get('sell', 0),
                            latest_rec.get('strongSell', 0)
                        ])
                        if total_ratings > 0:
                            analyst_score = (
                                (2 * latest_rec.get('strongBuy', 0) + latest_rec.get('buy', 0)) -
                                (2 * latest_rec.get('strongSell', 0) + latest_rec.get('sell', 0))
                            ) / (2 * total_ratings)
                            signals.append(1 if analyst_score > 0 else -1)
                            confidences.append(abs(analyst_score))
                            
                    # Get price target
                    price_target = await self.finnhub_client.get_price_target(ticker)
                    if price_target:
                        target_median = price_target.get('targetMedian')
                        if target_median and current_price:
                            price_signal = 1 if target_median > current_price else -1
                            price_confidence = min(abs(target_median - current_price) / current_price, 1.0)
                            signals.append(price_signal)
                            confidences.append(price_confidence)
                            
                except Exception as e:
                    logger.error(f"Failed to get Finnhub recommendations for {ticker}: {e}")
                
            # Initialize result dictionary
            result = {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'indicators': {
                    'price': current_price,
                    'sma_20': sma_20,
                    'rsi_14': rsi_14,
                    'macd': macd,
                    'signal_line': signal_line
                }
            }

            # Get news sentiment
            try:
                news_articles = await self.news_scraper.scrape_news()
                if news_articles:
                    # Calculate average sentiment for relevant news
                    relevant_articles = [
                        article for article in news_articles 
                        if ticker.lower() in article['title'].lower() or 
                           ticker.lower() in article['content'].lower()
                    ]
                    
                    if relevant_articles:
                        avg_sentiment = sum(float(article['sentiment_score']) for article in relevant_articles) / len(relevant_articles)
                        # Add sentiment signal
                        sentiment_signal = 1 if avg_sentiment > 0.2 else (-1 if avg_sentiment < -0.2 else 0)
                        sentiment_confidence = min(abs(avg_sentiment), 1.0)
                        signals.append(sentiment_signal)
                        confidences.append(sentiment_confidence)
                        
                        # Add news sentiment to result
                        result['news_sentiment'] = {
                            'score': avg_sentiment,
                            'article_count': len(relevant_articles)
                        }
            except Exception as e:
                logger.error(f"Failed to analyze news sentiment: {e}")

            # Combine signals with weighted sentiment
            signal_sum = sum(signals)
            if signal_sum > 0:
                signal = 'BUY'
            elif signal_sum < 0:
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'
                
            # Calculate overall confidence
            confidence = sum(c * abs(s) for c, s in zip(confidences, signals)) / len(signals)
            
            # Update result with final signal and confidence
            result.update({
                'signal': signal,
                'confidence': confidence,
                'indicators': {
                    'price': current_price,
                    'sma_20': sma_20,
                    'rsi_14': rsi_14,
                    'macd': macd,
                    'signal_line': signal_line
                }
            })
            
            # Add Finnhub data if available
            if self.finnhub_client:
                quote = await self.get_real_time_quote(ticker)
                if quote:
                    result['real_time'] = quote
                    
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze technicals for {ticker}: {e}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'indicators': {}
            }

if __name__ == '__main__':
    # Example usage
    import asyncio
    
    async def main():
        manager = MarketDataManager()
        analysis = await manager.analyze_technicals('AAPL')
        print(f"Technical Analysis for AAPL:")
        print(f"Signal: {analysis['signal']}")
        print(f"Confidence: {analysis['confidence']:.2%}")
        print("Indicators:", analysis['indicators'])
    
    asyncio.run(main())
