"""
Deep learning based trading strategy optimizer.
Uses transformer models to analyze patterns in market data and optimize trading decisions.
"""

import logging
from typing import Dict, List, Any, Optional
import torch
from torch import nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from src.data.market_data import MarketDataManager

logger = logging.getLogger(__name__)

class DeepTradeOptimizer:
    """
    Deep learning model for optimizing trading decisions based on market data,
    technical indicators, and sentiment analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize market data
        if config and 'market_data' in config:
            self.market_data = config['market_data']
        else:
            self.market_data = MarketDataManager()
        
        # Initialize the transformer model for sequence analysis
        self.model = self._initialize_model()
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        
        # Feature scaling
        self.feature_scaler = StandardScaler()
        
    def _initialize_model(self) -> nn.Module:
        """Initialize the deep learning model."""
        try:
            # Using ProsusAI's FinBERT model, which is specifically trained on financial text
            model = AutoModel.from_pretrained('ProsusAI/finbert')
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    async def prepare_market_features(
        self,
        ticker: str,
        current_suggestion: Optional[Dict[str, Any]] = None,
        lookback_days: int = 30
    ) -> np.ndarray:
        """Prepare market data features for the model."""
        try:
            # Get historical data and technical indicators
            tech_analysis = await self.market_data.analyze_technicals(ticker)
            
            # Extract relevant features
            features = {
                'price': tech_analysis['indicators']['price'],
                'volume': tech_analysis['indicators'].get('volume', 0),
                'rsi': tech_analysis['indicators'].get('rsi', 50),
                'macd': tech_analysis['indicators'].get('macd', 0),
                'bollinger_upper': tech_analysis['indicators'].get('bollinger_upper', 0),
                'bollinger_lower': tech_analysis['indicators'].get('bollinger_lower', 0)
            }
            
            # Convert to numpy array and scale
            feature_array = np.array(list(features.values())).reshape(1, -1)
            if not self.feature_scaler.n_samples_seen_:
                self.feature_scaler.fit(feature_array)
            scaled_features = self.feature_scaler.transform(feature_array)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Failed to prepare market features: {e}")
            return np.zeros((1, 6))  # Return zero features on error

    async def analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> torch.Tensor:
        """Analyze news sentiment using the transformer model."""
        try:
            if not articles:
                return torch.zeros((1, 768), device=self.device)  # Return zero embedding
                
            # Combine article titles and content
            texts = [f"{article['title']} {article['content']}" for article in articles]
            
            # Tokenize and encode texts
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Pool embeddings
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to analyze news sentiment: {e}")
            return torch.zeros((1, 768), device=self.device)

    async def optimize_trading_decision(
        self,
        ticker: str,
        news_data: List[Dict[str, Any]],
        current_suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize trading decisions by combining market data analysis,
        news sentiment, and current trading suggestions.
        """
        try:
            # Prepare features
            market_features = await self.prepare_market_features(ticker, current_suggestion)
            market_tensor = torch.FloatTensor(market_features).to(self.device)
            
            # Analyze news sentiment
            sentiment_embedding = await self.analyze_news_sentiment(news_data)
            
            # Combine features
            combined_features = torch.cat([
                market_tensor,
                sentiment_embedding
            ], dim=1)
            
            # Apply deep learning optimization
            with torch.no_grad():
                # Project features to trading decision space
                # Use linear projection for feature combination
                projection_layer = nn.Linear(combined_features.shape[1], 1).to(self.device)
                projected = projection_layer(combined_features)
                confidence_adjustment = torch.sigmoid(projected).item()
                
                # Adjust original trading suggestion
                optimized_suggestion = current_suggestion.copy()
                
                # Adjust confidence based on deep learning insights
                original_confidence = current_suggestion.get('confidence', 0.5)
                optimized_confidence = (
                    0.7 * original_confidence +  # Weight original analysis
                    0.3 * confidence_adjustment  # Weight AI adjustment
                )
                
                optimized_suggestion['confidence'] = optimized_confidence
                optimized_suggestion['ai_adjustment'] = confidence_adjustment
                
                # Add explanation of adjustment
                optimized_suggestion['optimization_notes'] = {
                    'market_signals': self._get_market_signal_strength(market_features),
                    'sentiment_strength': confidence_adjustment,
                    'confidence_adjustment': confidence_adjustment - original_confidence
                }
                
                return optimized_suggestion
                
        except Exception as e:
            logger.error(f"Failed to optimize trading decision: {e}")
            return current_suggestion  # Return original suggestion on error
            
    def _get_market_signal_strength(self, features: np.ndarray) -> float:
        """Calculate market signal strength from features."""
        try:
            # Normalize feature importance
            feature_weights = np.array([0.3, 0.1, 0.2, 0.2, 0.1, 0.1])  # Predefined weights
            signal_strength = np.abs(features * feature_weights).mean()
            return float(signal_strength)
        except Exception as e:
            logger.error(f"Failed to calculate market signal strength: {e}")
            return 0.5
