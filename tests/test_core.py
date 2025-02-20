"""
Core test suite for Trader Joe.
Tests basic functionality of main components.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import os
import tempfile
import json
import yaml
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.data.news_scraper import NewsScraper
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.engine.recommendation import RecommendationEngine, TradingSuggestion
from src.data.market_data import MarketDataManager

class TestConfiguration:
    """Test configuration that can be used across test cases."""
    
    @staticmethod
    def create_test_config():
        """Create a test configuration."""
        return {
            'trading': {
                'sentiment_threshold': 0.5,
                'max_recommendations': 5
            },
            'database': {
                'path': ':memory:'  # Use in-memory database for testing
            }
        }

class TestNewsScraper(unittest.IsolatedAsyncioTestCase):
    """Test the news scraping functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file for testing
        self.test_config = {
            'api_keys': {
                'newsapi': 'test-key',
                'finnhub': 'test-key'
            },
            'news_sources': [
                {'name': 'Test Source', 'url': 'https://test-finance.example.com/', 'priority': 1}
            ]
        }
        
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
            
        self.scraper = NewsScraper(config_path=self.config_path)
        
    @patch('aiohttp.ClientSession.get')
    async def test_scrape_news(self, mock_get):
        """Test scraping news articles."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = """
        <html>
            <div class="article">
                <h2>Test Article</h2>
                <p>Test content</p>
                <a href="/test-article">Read more</a>
            </div>
        </html>
        """
        mock_get.return_value.__aenter__.return_value = mock_response
        
        articles = await self.scraper.scrape_news()
        
        self.assertTrue(len(articles) > 0)
        article = articles[0]
        self.assertEqual(article['title'], 'Test Article')
        self.assertEqual(article['content'], 'Test content')
        self.assertTrue(article['url'].endswith('/test-article'))

class TestSentimentAnalysis(unittest.IsolatedAsyncioTestCase):
    """Test the sentiment analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = SentimentAnalyzer()
        
    async def test_analyze_sentiment_batch(self):
        """Test batch sentiment analysis."""
        texts = [
            "Great earnings report from the company!",
            "Disappointing results and outlook."
        ]
        
        results = await self.analyzer.analyze_sentiment_batch(texts)
        
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0]['sentiment_score'], 0)  # Positive sentiment
        self.assertLess(results[1]['sentiment_score'], 0)  # Negative sentiment

class TestRecommendationEngine(unittest.IsolatedAsyncioTestCase):
    """Test the recommendation engine functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = TestConfiguration.create_test_config()
        self.engine = RecommendationEngine()
        
    async def test_generate_suggestions(self):
        """Test generating trading suggestions."""
        # Test news data with sentiment
        news_data = [{
            'title': 'Test Article',
            'content': 'Company XYZ had great earnings',
            'url': 'http://test.com/article',
            'source': 'Test Source',
            'published_at': datetime.now().isoformat(),
            'sentiment_score': 0.8,
            'sentiment': 'positive'
        }]
        
        suggestions = await self.engine.generate_suggestions(news_data)
        
        # Should have at least one suggestion if sentiment is strong enough
        if suggestions:
            suggestion = suggestions[0]
            self.assertIsInstance(suggestion, TradingSuggestion)
            self.assertTrue(hasattr(suggestion, 'action'))
            self.assertTrue(hasattr(suggestion, 'confidence'))
            self.assertTrue(hasattr(suggestion, 'sentiment_score'))

if __name__ == '__main__':
    unittest.main()
