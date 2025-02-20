"""
Web interface for Trader Joe.
Provides a dashboard for viewing trading suggestions and news.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, cast
import yaml
import sqlite3
import asyncio
import time
import json
from datetime import datetime, timedelta
from functools import wraps

# Flask imports
from flask import Flask, render_template, jsonify, send_from_directory, url_for, redirect, request
from flask_socketio import SocketIO, emit

# Dash imports
import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

# Stock data
import yfinance as yf
import ta

# Import project modules
from src.engine.recommendation import RecommendationEngine, TradingSuggestion
from src.data.news_scraper import NewsScraper
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.web.assets.layout import create_layout
from src.web.assets.enhanced_dashboard import create_market_overview
from src.data import init_database
from src.data.models import (
    init_db,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    get_portfolio,
    add_to_portfolio,
    update_portfolio_holding,
    remove_from_portfolio,
    set_price_alert,
    get_active_alerts,
    mark_alert_triggered,
    get_user_preferences,
    update_user_preferences
)

logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for a given DataFrame."""
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain = gain.apply(lambda x: max(x, 0))
    loss = loss.apply(lambda x: max(-x, 0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)
    
    return df

def async_callback(func):
    """Decorator to handle async callbacks in Dash."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

def get_stock_info(symbol: str) -> Dict[str, Any]:
    """Get current stock information."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'symbol': symbol,
            'price': info.get('regularMarketPrice', 0),
            'change': info.get('regularMarketChange', 0),
            'change_percent': info.get('regularMarketChangePercent', 0),
            'volume': info.get('regularMarketVolume', 0),
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown'),
            'name': info.get('longName', symbol)
        }
    except Exception as e:
        logging.error(f"Error fetching stock info for {symbol}: {e}")
        return {
            'symbol': symbol,
            'price': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'market_cap': 0,
            'sector': 'Unknown',
            'name': symbol
        }

async def get_news_with_sentiment() -> List[Dict[str, Any]]:
    """Get news articles with sentiment analysis."""
    news_scraper = NewsScraper()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Scrape news
    news_data = await news_scraper.scrape_news()
    
    # Extract content for sentiment analysis
    news_contents = [article['content'] for article in news_data]
    
    # Analyze sentiment in batch
    sentiment_results = await sentiment_analyzer.analyze_sentiment_batch(news_contents)
    
    # Combine news data with sentiment results
    news_with_sentiment = []
    for article, sentiment in zip(news_data, sentiment_results):
        article_with_sentiment = article.copy()
        article_with_sentiment.update(sentiment)
        news_with_sentiment.append(article_with_sentiment)
        
    return news_with_sentiment

def create_app(config: Optional[Dict[str, Any]] = None) -> Tuple[Flask, SocketIO]:
    """Create and configure the Flask application."""
    static_folder = str(Path(__file__).parent / 'static')
    template_folder = str(Path(__file__).parent / 'templates')
    
    server = Flask(__name__, 
                  static_folder=static_folder,
                  template_folder=template_folder)
    server.config['SECRET_KEY'] = 'trader-joe-secret-key'
    
    if config:
        server.config['DATABASE_PATH'] = config['database']['path']
        init_database(config['database']['path'])
    
    # Initialize SocketIO
    socketio = SocketIO(server, cors_allowed_origins="*", async_mode='threading')
    
    # Routes
    @server.route('/')
    def index():
        """Render the main dashboard page."""
        return redirect('/dashboard/')

    @server.route('/api/suggestions')
    async def get_suggestions():
        """API endpoint for getting current trading suggestions."""
        if not config:
            return jsonify([])
            
        try:
            # Get news with sentiment
            news_with_sentiment = await get_news_with_sentiment()
            
            # Initialize engine and get suggestions
            engine = RecommendationEngine(config_path='config/config.yaml')
            suggestions = await engine.generate_suggestions(news_with_sentiment)
            
            return jsonify([s.to_dict() for s in suggestions])
            
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return jsonify({'error': str(e)}), 500

    # Watchlist API routes
    @server.route('/api/watchlist', methods=['GET'])
    def get_watchlist_api():
        """Get watchlist items."""
        try:
            watchlist = get_watchlist(server.config['DATABASE_PATH'])
            items = []
            for item in watchlist:
                info = get_stock_info(item.symbol)
                items.append({
                    'id': item.id,
                    'symbol': item.symbol,
                    'price': info['price'],
                    'change': info['change'],
                    'change_percent': info['change_percent'],
                    'added_at': item.added_at.isoformat(),
                    'notes': item.notes
                })
            return jsonify(items)
        except Exception as e:
            logger.error(f"Failed to get watchlist: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/watchlist', methods=['POST'])
    def add_to_watchlist_api():
        """Add stock to watchlist."""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            notes = data.get('notes')
            
            if not symbol:
                return jsonify({'error': 'Symbol is required'}), 400
                
            item = add_to_watchlist(server.config['DATABASE_PATH'], symbol, notes)
            return jsonify({
                'id': item.id,
                'symbol': item.symbol,
                'added_at': item.added_at.isoformat(),
                'notes': item.notes
            })
        except Exception as e:
            logger.error(f"Failed to add to watchlist: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/watchlist/<symbol>', methods=['DELETE'])
    def remove_from_watchlist_api(symbol):
        """Remove stock from watchlist."""
        try:
            success = remove_from_watchlist(server.config['DATABASE_PATH'], symbol)
            if success:
                return jsonify({'message': f'Removed {symbol} from watchlist'})
            return jsonify({'error': 'Symbol not found in watchlist'}), 404
        except Exception as e:
            logger.error(f"Failed to remove from watchlist: {e}")
            return jsonify({'error': str(e)}), 500

    # Portfolio API routes
    @server.route('/api/portfolio', methods=['GET'])
    def get_portfolio_api():
        """Get portfolio holdings."""
        try:
            holdings = get_portfolio(server.config['DATABASE_PATH'])
            items = []
            for holding in holdings:
                info = get_stock_info(holding.symbol)
                current_value = holding.quantity * info['price']
                cost_basis = holding.quantity * holding.average_cost
                gain_loss = current_value - cost_basis
                gain_loss_percent = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                
                items.append({
                    'id': holding.id,
                    'symbol': holding.symbol,
                    'quantity': holding.quantity,
                    'average_cost': holding.average_cost,
                    'current_price': info['price'],
                    'current_value': current_value,
                    'gain_loss': gain_loss,
                    'gain_loss_percent': gain_loss_percent,
                    'purchase_date': holding.purchase_date.isoformat(),
                    'notes': holding.notes
                })
            return jsonify(items)
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/portfolio', methods=['POST'])
    def add_to_portfolio_api():
        """Add position to portfolio."""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            quantity = data.get('quantity')
            average_cost = data.get('average_cost')
            notes = data.get('notes')
            
            if not all([symbol, quantity, average_cost]):
                return jsonify({'error': 'Symbol, quantity, and average cost are required'}), 400
                
            holding = add_to_portfolio(
                server.config['DATABASE_PATH'],
                symbol,
                float(quantity),
                float(average_cost),
                notes
            )
            return jsonify({
                'id': holding.id,
                'symbol': holding.symbol,
                'quantity': holding.quantity,
                'average_cost': holding.average_cost,
                'purchase_date': holding.purchase_date.isoformat(),
                'notes': holding.notes
            })
        except Exception as e:
            logger.error(f"Failed to add to portfolio: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/portfolio/<symbol>', methods=['PUT'])
    def update_portfolio_api(symbol):
        """Update portfolio position."""
        try:
            data = request.get_json()
            quantity = data.get('quantity')
            average_cost = data.get('average_cost')
            
            if not all([quantity, average_cost]):
                return jsonify({'error': 'Quantity and average cost are required'}), 400
                
            success = update_portfolio_holding(
                server.config['DATABASE_PATH'],
                symbol,
                float(quantity),
                float(average_cost)
            )
            if success:
                return jsonify({'message': f'Updated {symbol} position'})
            return jsonify({'error': 'Symbol not found in portfolio'}), 404
        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/portfolio/<symbol>', methods=['DELETE'])
    def remove_from_portfolio_api(symbol):
        """Remove position from portfolio."""
        try:
            success = remove_from_portfolio(server.config['DATABASE_PATH'], symbol)
            if success:
                return jsonify({'message': f'Removed {symbol} from portfolio'})
            return jsonify({'error': 'Symbol not found in portfolio'}), 404
        except Exception as e:
            logger.error(f"Failed to remove from portfolio: {e}")
            return jsonify({'error': str(e)}), 500

    # Price Alerts API routes
    @server.route('/api/alerts', methods=['GET'])
    def get_alerts_api():
        """Get active price alerts."""
        try:
            alerts = get_active_alerts(server.config['DATABASE_PATH'])
            return jsonify(alerts)
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/alerts', methods=['POST'])
    def set_alert_api():
        """Set new price alert."""
        try:
            data = request.get_json()
            symbol = data.get('symbol')
            target_price = data.get('target_price')
            is_above = data.get('is_above', True)
            
            if not all([symbol, target_price]):
                return jsonify({'error': 'Symbol and target price are required'}), 400
                
            alert_id = set_price_alert(
                server.config['DATABASE_PATH'],
                symbol,
                float(target_price),
                bool(is_above)
            )
            return jsonify({'id': alert_id})
        except Exception as e:
            logger.error(f"Failed to set alert: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/alerts/<int:alert_id>/trigger', methods=['POST'])
    def trigger_alert_api(alert_id):
        """Mark alert as triggered."""
        try:
            mark_alert_triggered(server.config['DATABASE_PATH'], alert_id)
            return jsonify({'message': f'Marked alert {alert_id} as triggered'})
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
            return jsonify({'error': str(e)}), 500

    # User Preferences API routes
    @server.route('/api/preferences', methods=['GET'])
    def get_preferences_api():
        """Get user preferences."""
        try:
            preferences = get_user_preferences(server.config['DATABASE_PATH'])
            return jsonify(preferences)
        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/preferences', methods=['PUT'])
    def update_preferences_api():
        """Update user preferences."""
        try:
            preferences = request.get_json()
            success = update_user_preferences(server.config['DATABASE_PATH'], preferences)
            if success:
                return jsonify({'message': 'Updated preferences'})
            return jsonify({'error': 'Failed to update preferences'}), 500
        except Exception as e:
            logger.error(f"Failed to update preferences: {e}")
            return jsonify({'error': str(e)}), 500

    @server.route('/api/news')
    def get_news():
        """API endpoint for getting recent news articles."""
        try:
            if 'DATABASE_PATH' not in server.config:
                return jsonify([])
                
            with sqlite3.connect(server.config['DATABASE_PATH']) as conn:
                cursor = conn.execute('''
                    SELECT title, content, source, url, published_at, sentiment_score
                    FROM articles
                    ORDER BY published_at DESC
                    LIMIT 50
                ''')
                
                articles = [{
                    'title': row[0],
                    'content': row[1],
                    'source': row[2],
                    'url': row[3],
                    'published_at': row[4],
                    'sentiment_score': row[5]
                } for row in cursor.fetchall()]
                
                return jsonify(articles)
                
        except Exception as e:
            logger.error(f"Failed to get news articles: {e}")
            return jsonify({'error': str(e)}), 500
            
    return server, socketio

def create_dashboard(config: Dict[str, Any], server: Flask) -> dash.Dash:
    """Create and configure the Dash dashboard."""
    # Initialize Dash
    assets_folder = str(Path(__file__).parent / 'static')
    dash_app = dash.Dash(
        __name__,
        server=server,
        assets_folder=assets_folder,
        external_stylesheets=[
            'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
            '/dashboard/assets/css/dashboard.css'
        ],
        url_base_pathname='/dashboard/',
        serve_locally=True,
        title='Trader Joe Dashboard'
    )

    # Initialize recommendation engine
    recommendation_engine = RecommendationEngine(config_path='config/config.yaml')

    # Setup dashboard layout with optimized update intervals
    update_intervals = {
        'market_data': config['web']['chart_update_interval'] * 1000,  # Convert to milliseconds
        'watchlist': config['web']['chart_update_interval'] * 2000,    # Less frequent updates
        'news': config['web']['chart_update_interval'] * 5000,         # News updates every 5x interval
        'suggestions': config['web']['chart_update_interval'] * 10000   # Suggestions every 10x interval
    }
    dash_app.layout = create_layout(update_intervals)

    # Setup callbacks
    @dash_app.callback(
        Output('sentiment-chart', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    @async_callback
    async def update_sentiment_chart(_):
        """Update the sentiment overview chart."""
        news_with_sentiment = await get_news_with_sentiment()
        suggestions = await recommendation_engine.generate_suggestions(news_with_sentiment)
        
        if not suggestions:
            return go.Figure()
            
        df = pd.DataFrame([s.to_dict() for s in suggestions])
        
        return {
            'data': [
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sentiment_score'],
                    mode='markers',
                    marker={
                        'size': df['confidence'] * 20,
                        'color': ['#ef5350' if x == 'SELL' else '#66bb6a' for x in df['action']],
                        'line': {'width': 1, 'color': '#ffffff'}
                    },
                    text=df['ticker'],
                    hovertemplate=(
                        '<b>%{text}</b><br>' +
                        'Sentiment: %{y:.2f}<br>' +
                        'Confidence: %{marker.size:.2f}<br>' +
                        'Time: %{x}<br>' +
                        '<extra></extra>'
                    )
                )
            ],
            'layout': go.Layout(
                title={
                    'text': 'Market Sentiment Trends',
                    'font': {'color': '#ffffff', 'size': 20}
                },
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                xaxis={
                    'title': 'Time',
                    'gridcolor': '#2c2c2c',
                    'zerolinecolor': '#2c2c2c',
                    'tickfont': {'color': '#ffffff'},
                    'titlefont': {'color': '#ffffff'}
                },
                yaxis={
                    'title': 'Sentiment Score',
                    'gridcolor': '#2c2c2c',
                    'zerolinecolor': '#2c2c2c',
                    'tickfont': {'color': '#ffffff'},
                    'titlefont': {'color': '#ffffff'}
                },
                hovermode='closest',
                margin={'t': 50, 'b': 50, 'l': 50, 'r': 20}
            )
        }

    @dash_app.callback(
        Output('ticker-selector', 'options'),
        Output('ticker-selector', 'value'),
        Input('interval-component', 'n_intervals')
    )
    @async_callback
    async def update_ticker_dropdown(_):
        """Update the ticker selection dropdown."""
        news_with_sentiment = await get_news_with_sentiment()
        suggestions = await recommendation_engine.generate_suggestions(news_with_sentiment)
        
        if not suggestions:
            return [], None
            
        tickers = sorted(set(s.ticker for s in suggestions))
        options = [{'label': t, 'value': t} for t in tickers]
        
        return options, options[0]['value'] if options else None

    @dash_app.callback(
        Output('price-chart', 'figure'),
        Input('ticker-selector', 'value'),
        Input('interval-component', 'n_intervals')
    )
    def update_price_chart(ticker: Optional[str], _):
        """Update the price history chart."""
        if not ticker:
            return go.Figure()
            
        # Get price history from yfinance
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period='1d', interval='5m')
            
            return {
                'data': [
                    go.Candlestick(
                        x=history.index,
                        open=history['Open'],
                        high=history['High'],
                        low=history['Low'],
                        close=history['Close'],
                        increasing={'line': {'color': '#66bb6a'}},
                        decreasing={'line': {'color': '#ef5350'}}
                    )
                ],
                'layout': go.Layout(
                    title={
                        'text': f'{ticker} Price History',
                        'font': {'color': '#ffffff', 'size': 20}
                    },
                    paper_bgcolor='#1e1e1e',
                    plot_bgcolor='#1e1e1e',
                    xaxis={
                        'title': 'Time',
                        'gridcolor': '#2c2c2c',
                        'zerolinecolor': '#2c2c2c',
                        'tickfont': {'color': '#ffffff'},
                        'titlefont': {'color': '#ffffff'}
                    },
                    yaxis={
                        'title': 'Price ($)',
                        'gridcolor': '#2c2c2c',
                        'zerolinecolor': '#2c2c2c',
                        'tickfont': {'color': '#ffffff'},
                        'titlefont': {'color': '#ffffff'}
                    },
                    height=400,
                    margin={'t': 50, 'b': 50, 'l': 50, 'r': 20}
                )
            }
        except Exception as e:
            logger.error(f"Failed to get price history for {ticker}: {e}")
            return go.Figure()

    @dash_app.callback(
        Output('suggestions-container', 'children'),
        Input('interval-component', 'n_intervals')
    )
    @async_callback
    async def update_suggestions(_):
        """Update the trading suggestions panel."""
        news_with_sentiment = await get_news_with_sentiment()
        suggestions = await recommendation_engine.generate_suggestions(news_with_sentiment)
        
        if not suggestions:
            return html.P("No current trading suggestions.")
            
        return [
            html.Div([
                html.H4(f"{s.ticker} - {s.action}"),
                html.P([
                    f"Confidence: {s.confidence:.2%}",
                    html.Br(),
                    f"Sentiment: {s.sentiment_score:.2f}",
                    html.Br(),
                    f"Price: ${s.price:.2f}"
                ]),
                html.Div([
                    html.P("Supporting Articles:"),
                    html.Ul([
                        html.Li(article['title'])
                        for article in s.supporting_articles[:3]
                    ])
                ], className='supporting-articles')
            ], className=f"suggestion-card {s.action.lower()}")
            for s in suggestions
        ]

    # Theme toggle callback
    @dash_app.callback(
        Output("dashboard-container", "data-theme"),
        Input("theme-toggle", "n_clicks"),
        State("dashboard-container", "data-theme")
    )
    def toggle_theme(n_clicks, current_theme):
        """Toggle between light and dark theme."""
        if n_clicks is None:
            return current_theme or "dark"
        return "light" if current_theme == "dark" else "dark"

    # Watchlist callbacks
    @dash_app.callback(
        Output("watchlist-content", "children"),
        Input("interval-component", "n_intervals"),
        State("watchlist-store", "data")
    )
    def update_watchlist(_, stored_watchlist):
        """Update watchlist display."""
        try:
            watchlist = get_watchlist(config['database']['path'])
            
            if not watchlist:
                return html.P("No stocks in watchlist.")
            
            stock_items = []
            for item in watchlist:
                info = get_stock_info(item.symbol)
                stock_items.append(
                    html.Div([
                        html.Div([
                            html.Span(info['symbol'], className="stock-symbol"),
                            html.Span(f"${info['price']:.2f}", className="stock-price")
                        ], className="stock-info"),
                        html.Div([
                            html.Span(
                                f"{info['change_percent']:.1f}%",
                                className=f"stock-change {'positive' if info['change'] >= 0 else 'negative'}"
                            ),
                            html.Button(
                                "🔔",
                                id={"type": "alert-btn", "symbol": info['symbol']},
                                className="action-btn ml-2",
                                title="Set alert"
                            ),
                            html.Button(
                                "✖",
                                id={"type": "remove-watchlist-btn", "symbol": info['symbol']},
                                className="action-btn ml-2",
                                title="Remove from watchlist"
                            )
                        ], className="stock-actions")
                    ], className="stock-item")
                )
            
            return stock_items
            
        except Exception as e:
            logger.error(f"Failed to update watchlist: {e}")
            return html.P("Error loading watchlist.")

    # Portfolio callbacks
    @dash_app.callback(
        [Output("portfolio-content", "children"),
         Output("portfolio-value", "children"),
         Output("portfolio-change", "children")],
        Input("interval-component", "n_intervals"),
        State("portfolio-store", "data")
    )
    def update_portfolio(_, stored_portfolio):
        """Update portfolio display."""
        try:
            holdings = get_portfolio(config['database']['path'])
            
            if not holdings:
                return html.P("No stocks in portfolio."), "$0.00", "0.00%"
            
            total_value = 0
            total_cost = 0
            stock_items = []
            
            for holding in holdings:
                info = get_stock_info(holding.symbol)
                current_value = holding.quantity * info['price']
                total_value += current_value
                total_cost += holding.quantity * holding.average_cost
                
                stock_items.append(
                    html.Div([
                        html.Div([
                            html.Span(info['symbol'], className="stock-symbol"),
                            html.Div([
                                html.Span(f"{holding.quantity:.2f} shares @ ${holding.average_cost:.2f}", className="position-details"),
                                html.Span(f"Current: ${info['price']:.2f}", className="current-price")
                            ])
                        ], className="stock-info"),
                        html.Div([
                            html.Span(
                                f"${current_value:.2f}",
                                className="position-value"
                            ),
                            html.Button(
                                "✎",
                                id={"type": "edit-position-btn", "symbol": info['symbol']},
                                className="action-btn ml-2",
                                title="Edit position"
                            ),
                            html.Button(
                                "✖",
                                id={"type": "remove-position-btn", "symbol": info['symbol']},
                                className="action-btn ml-2",
                                title="Remove position"
                            )
                        ], className="stock-actions")
                    ], className="stock-item")
                )
            
            total_change = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
            change_class = "positive" if total_change >= 0 else "negative"
            
            return (
                stock_items,
                f"${total_value:.2f}",
                html.Span(f"{total_change:+.2f}%", className=f"portfolio-change {change_class}")
            )
            
        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")
            return html.P("Error loading portfolio."), "$0.00", "0.00%"

    # Real-time ticker callback
    @dash_app.callback(
        Output("ticker-content", "children"),
        Input("interval-component", "n_intervals")
    )
    def update_ticker(_):
        """Update the ticker tape with major indices and watchlist stocks."""
        try:
            # Major indices
            indices = ['SPY', 'QQQ', 'DIA', 'IWM']
            ticker_items = []
            
            for symbol in indices:
                info = get_stock_info(symbol)
                ticker_items.append(
                    html.Span([
                        f"{symbol} ${info['price']:.2f} ",
                        html.Span(
                            f"{info['change_percent']:+.1f}%",
                            className=f"{'positive' if info['change'] >= 0 else 'negative'}"
                        )
                    ], className="ticker-item")
                )
            
            # Add watchlist stocks
            watchlist = get_watchlist(config['database']['path'])
            for item in watchlist:
                info = get_stock_info(item.symbol)
                ticker_items.append(
                    html.Span([
                        f"{info['symbol']} ${info['price']:.2f} ",
                        html.Span(
                            f"{info['change_percent']:+.1f}%",
                            className=f"{'positive' if info['change'] >= 0 else 'negative'}"
                        )
                    ], className="ticker-item")
                )
            
            return ticker_items
            
        except Exception as e:
            logger.error(f"Failed to update ticker: {e}")
            return []

    # Search and filter callbacks
    @dash_app.callback(
        Output("ticker-selector", "options"),
        [Input("stock-search", "value"),
         Input("sector-filter", "value"),
         Input("market-cap-filter", "value"),
         Input("price-range-filter", "value")]
    )
    def filter_stocks(search_term, sector, market_cap, price_range):
        """Filter stocks based on search and filter criteria."""
        try:
            # Get all stocks from watchlist and portfolio
            watchlist = get_watchlist(config['database']['path'])
            portfolio = get_portfolio(config['database']['path'])
            symbols = set(item.symbol for item in watchlist + portfolio)
            
            # Add major indices and popular stocks
            symbols.update(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA'])
            
            filtered_stocks = []
            for symbol in symbols:
                info = get_stock_info(symbol)
                
                # Apply filters
                if search_term and search_term.upper() not in symbol:
                    continue
                    
                if sector and info['sector'] != sector:
                    continue
                    
                if market_cap:
                    cap = info['market_cap']
                    if market_cap == 'large' and cap < 10e9:
                        continue
                    elif market_cap == 'mid' and (cap < 2e9 or cap > 10e9):
                        continue
                    elif market_cap == 'small' and cap > 2e9:
                        continue
                
                if price_range:
                    if not (price_range[0] <= info['price'] <= price_range[1]):
                        continue
                
                filtered_stocks.append({
                    'label': f"{symbol} - {info['name']}",
                    'value': symbol
                })
            
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"Failed to filter stocks: {e}")
            return []

    # Technical indicators callback
    @dash_app.callback(
        Output('price-chart', 'figure'),
        [Input('ticker-selector', 'value'),
         Input('technical-indicators', 'value'),
         Input('timeframe-selector', 'value')]
    )
    def update_price_chart_with_indicators(ticker: Optional[str], indicators: List[str], timeframe: str):
        """Update price chart with selected technical indicators."""
        if not ticker:
            return go.Figure()
            
        try:
            # Get price history from yfinance
            stock = yf.Ticker(ticker)
            
            # Map timeframe to period and interval
            timeframe_map = {
                '1D': ('1d', '5m'),
                '1W': ('5d', '15m'),
                '1M': ('1mo', '1h'),
                '3M': ('3mo', '1d'),
                '1Y': ('1y', '1d')
            }
            period, interval = timeframe_map.get(timeframe, ('1d', '5m'))
            
            history = stock.history(period=period, interval=interval)
            
            if indicators:
                history = calculate_technical_indicators(history)
            
            # Create subplots if needed
            fig = make_subplots(
                rows=3 if 'RSI' in (indicators or []) else 2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2] if 'RSI' in (indicators or []) else [0.7, 0.3]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=history.index,
                    open=history['Open'],
                    high=history['High'],
                    low=history['Low'],
                    close=history['Close'],
                    name='Price',
                    increasing={'line': {'color': '#66bb6a'}},
                    decreasing={'line': {'color': '#ef5350'}}
                ),
                row=1, col=1
            )
            
            # Add selected indicators
            if indicators:
                if 'MA' in indicators:
                    fig.add_trace(
                        go.Scatter(
                            x=history.index,
                            y=history['MA20'],
                            name='MA20',
                            line={'color': '#42a5f5'}
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=history.index,
                            y=history['MA50'],
                            name='MA50',
                            line={'color': '#ab47bc'}
                        ),
                        row=1, col=1
                    )
                
                if 'BB' in indicators:
                    fig.add_trace(
                        go.Scatter(
                            x=history.index,
                            y=history['BB_upper'],
                            name='BB Upper',
                            line={'color': '#90a4ae', 'dash': 'dash'}
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=history.index,
                            y=history['BB_lower'],
                            name='BB Lower',
                            line={'color': '#90a4ae', 'dash': 'dash'},
                            fill='tonexty'
                        ),
                        row=1, col=1
                    )
                
                if 'MACD' in indicators:
                    fig.add_trace(
                        go.Bar(
                            x=history.index,
                            y=history['MACD'] - history['MACD_signal'],
                            name='MACD Histogram',
                            marker={'color': '#90a4ae'}
                        ),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=history.index,
                            y=history['MACD'],
                            name='MACD',
                            line={'color': '#42a5f5'}
                        ),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=history.index,
                            y=history['MACD_signal'],
                            name='Signal',
                            line={'color': '#ef5350'}
                        ),
                        row=2, col=1
                    )
                
                if 'RSI' in indicators:
                    fig.add_trace(
                        go.Scatter(
                            x=history.index,
                            y=history['RSI'],
                            name='RSI',
                            line={'color': '#42a5f5'}
                        ),
                        row=3, col=1
                    )
                    # Add overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row="3", col="1")
                    fig.add_hline(y=30, line_dash="dash", line_color="#66bb6a", row="3", col="1")
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f'{ticker} Price History',
                    'font': {'color': '#ffffff', 'size': 20}
                },
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                xaxis={'gridcolor': '#2c2c2c', 'zerolinecolor': '#2c2c2c'},
                yaxis={'gridcolor': '#2c2c2c', 'zerolinecolor': '#2c2c2c'},
                font={'color': '#ffffff'},
                showlegend=True,
                height=800,
                margin={'t': 50, 'b': 50, 'l': 50, 'r': 20},
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to update price chart: {e}")
            return go.Figure()

    @dash_app.callback(
        Output('news-feed', 'children'),
        [Input('interval-component', 'n_intervals'),
         Input('news-filter', 'value')]
    )
    @async_callback
    async def update_news_feed(_, filter_type):
        """Update the news feed panel with filtered news."""
        try:
            news_with_sentiment = await get_news_with_sentiment()
            
            if not news_with_sentiment:
                return html.P("No recent news articles.")
            
            # Filter news based on selection
            if filter_type in ['watchlist', 'portfolio']:
                symbols = set()
                if filter_type == 'watchlist':
                    watchlist = get_watchlist(config['database']['path'])
                    symbols = set(item.symbol for item in watchlist)
                else:
                    portfolio = get_portfolio(config['database']['path'])
                    symbols = set(item.symbol for item in portfolio)
                
                filtered_news = [
                    article for article in news_with_sentiment
                    if any(symbol in article['title'] for symbol in symbols)
                ]
            else:
                filtered_news = news_with_sentiment
            
            return [
                html.Div([
                    html.Span(article['source'], className="news-source"),
                    html.Span(
                        f"{article['sentiment_score']:.1f}",
                        className=f"news-sentiment {'sentiment-positive' if article['sentiment_score'] > 0 else 'sentiment-negative' if article['sentiment_score'] < 0 else 'sentiment-neutral'}"
                    ),
                    html.H4(article['title']),
                    html.P([
                        article['content'][:200] + "...",
                        html.Br(),
                        html.Small(f"Published: {article['published_at']}")
                    ])
                ], className='news-card')
                for article in filtered_news[:config['web']['max_displayed_news']]
            ]
                
        except Exception as e:
            logger.error(f"Failed to update news feed: {e}")
            return html.P("Error loading news feed.")

    # Modal callbacks
    @dash_app.callback(
        [Output('add-stock-modal', 'className'),
         Output('add-stock-modal-title', 'children'),
         Output('portfolio-fields', 'style')],
        [Input('add-to-watchlist-btn', 'n_clicks'),
         Input('close-add-stock-modal', 'n_clicks')],
        [State('modal-store', 'data')]
    )
    def toggle_add_stock_modal(open_clicks, close_clicks, modal_data):
        """Toggle the add stock modal."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return "modal fade", "Add to Watchlist", {'display': 'none'}
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'add-to-watchlist-btn':
            return "modal fade show d-block", "Add to Watchlist", {'display': 'none'}
        elif trigger_id == 'close-add-stock-modal':
            return "modal fade", "Add to Watchlist", {'display': 'none'}
            
        return "modal fade", "Add to Watchlist", {'display': 'none'}

    @dash_app.callback(
        [Output('watchlist-store', 'data'),
         Output('add-stock-modal', 'style', allow_duplicate=True)],
        [Input('confirm-add-stock', 'n_clicks')],
        [State('add-stock-symbol', 'value'),
         State('add-stock-notes', 'value'),
         State('watchlist-store', 'data')],
        prevent_initial_call=True
    )
    def add_stock_to_watchlist(n_clicks, symbol, notes, current_watchlist):
        """Add a stock to the watchlist."""
        if not n_clicks or not symbol:
            raise dash.exceptions.PreventUpdate
            
        try:
            item = add_to_watchlist(config['database']['path'], symbol.upper(), notes)
            return current_watchlist or [], {'display': 'none'}
        except Exception as e:
            logger.error(f"Failed to add stock to watchlist: {e}")
            raise dash.exceptions.PreventUpdate

    return dash_app

def init_app(config: Dict[str, Any]):
    """Initialize the complete application."""
    # Initialize database
    if config and 'database' in config:
        init_db(config['database']['path'])
    
    # Create Flask server and SocketIO
    server, socketio = create_app(config)
    
    # Create Dash app
    dash_app = create_dashboard(config, server)
    
    # Setup background task for suggestions
    def update_loop():
        """Background task to generate suggestions."""
        while True:
            try:
                # Create event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Get news with sentiment
                news_with_sentiment = loop.run_until_complete(get_news_with_sentiment())
                
                # Generate new suggestions
                engine = RecommendationEngine(config_path='config/config.yaml')
                suggestions = loop.run_until_complete(engine.generate_suggestions(news_with_sentiment))
                
                # Emit updates via WebSocket
                if suggestions:
                    socketio.emit('suggestions_update', {
                        'suggestions': [s.to_dict() for s in suggestions]
                    })
                    
                # Sleep until next update
                loop.run_until_complete(asyncio.sleep(config['web']['chart_update_interval']))
                
            except Exception as e:
                logger.error(f"Update thread error: {e}")
                time.sleep(5)  # Sleep on error to prevent rapid retries
            finally:
                loop.close()

    # Start background task when client connects
    @socketio.on('connect')
    def handle_connect():
        socketio.start_background_task(target=update_loop)
    
    return server, socketio, dash_app

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize application
    server, socketio, dash_app = init_app(config)
    
    # Run the server
    socketio.run(
        server,
        host=config['web']['host'],
        port=config['web']['port'],
        debug=config['web']['debug']
    )
