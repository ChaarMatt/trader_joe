"""
Sentiment analysis functionality for Trader Joe.
"""

import logging
from typing import List, Dict, Any
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio
from src.utils.text_processing import clean_text

nltk.download('vader_lexicon')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes sentiment of text using VADER sentiment analysis."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a given text."""
        try:
            sentiment = self.sia.polarity_scores(text)
            return {
                'text': clean_text(text),
                'sentiment_score': sentiment['compound'],
                'sentiment': 'positive' if sentiment['compound'] >= 0.05 else 
                            'negative' if sentiment['compound'] <= -0.05 else 'neutral'
            }
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return {'text': text, 'sentiment_score': 0, 'sentiment': 'unknown'}

    async def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment of multiple texts."""
        try:
            results = []
            for text in texts:
                result = await self.analyze_sentiment(text)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Failed to analyze sentiment batch: {e}")
            return []

if __name__ == '__main__':
    # Example usage
    analyzer = SentimentAnalyzer()
    result = asyncio.run(analyzer.analyze_sentiment("I love this stock!"))
    print(f"Sentiment: {result['sentiment']}, Score: {result['sentiment_score']}")
