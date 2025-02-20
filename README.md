# Trader Joe

A sophisticated trading bot that leverages machine learning, sentiment analysis, and technical indicators to generate intelligent trading recommendations.

## Core Components

### Data Layer (`src/data/`)
- **Market Data** (`market_data.py`): Real-time market data integration via Finnhub
- **News Scraping** (`news_scraper.py`): Multi-source financial news aggregation
- **Rate Limiting** (`rate_limiter.py`): Smart API request management
- **Data Models** (`models.py`): SQLAlchemy models for data persistence
- **Ticker Management** (`ticker_mapper.py`): Symbol mapping and validation

### Analysis Engine (`src/analysis/`)
- **Deep Optimizer** (`deep_optimizer.py`): Machine learning model for pattern recognition and trade optimization
- **Sentiment Analysis** (`sentiment_analysis.py`): Natural language processing for news sentiment scoring

### Trading Engine (`src/engine/`)
- **Recommendation System** (`recommendation.py`): Combines technical indicators, sentiment analysis, and ML predictions

### Web Interface (`src/web/`)
- **Dashboard** (`app.py`): Flask-based real-time trading dashboard
- **Layouts** (`assets/layout.py`): Responsive UI components
- **Templates** (`templates/`): HTML templates for web interface
- **Static Assets** (`static/`): CSS, JavaScript, and other static resources

## Key Features

### Market Analysis
- Real-time market data streaming
- Technical indicator calculations
- Volume analysis and trend detection
- Price movement pattern recognition

### News Processing
- Multi-source financial news aggregation
- Real-time sentiment analysis
- Impact scoring for news events
- Historical sentiment tracking

### Machine Learning
- Deep learning for pattern recognition
- Automated parameter optimization
- Market regime detection
- Risk-adjusted return optimization

### Trading Logic
- Multi-factor recommendation engine
- Risk management rules
- Position sizing optimization
- Entry/exit timing optimization

### Web Dashboard
- Real-time market data visualization
- Trading recommendations display
- Performance metrics tracking
- News sentiment overview

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys:
- Copy `config/config.example.yaml` to `config/config.yaml`
- Add your API keys for:
  - NewsAPI (https://newsapi.org/)
  - Finnhub (https://finnhub.io/)

3. Initialize the database:
```bash
python src/main.py --init-db
```

## Usage

Start the trading bot:
```bash
python src/main.py
```

Access the dashboard:
- Open http://localhost:5000 in your browser
- View real-time trading recommendations
- Monitor market data and news sentiment
- Track technical indicators and ML predictions

## Configuration

The `config/config.yaml` file allows customization of:

### API Settings
- API keys and endpoints
- Rate limiting parameters
- Request timeout settings

### Trading Parameters
- Technical indicator settings
- Position sizing rules
- Risk management thresholds
- Trading time windows

### Analysis Settings
- Sentiment analysis weights
- ML model parameters
- Pattern recognition thresholds
- Risk tolerance levels

### Web Interface
- Update intervals
- Display preferences
- Chart configurations
- Performance metrics

## Development

Run the test suite:
```bash
python -m pytest tests/
```

### Test Coverage
- Core functionality tests
- Market data integration
- Ticker mapping validation
- Recommendation engine logic

## Logging

Comprehensive logging system:
- Location: `logs/trader_joe.log`
- Configurable log levels
- Rotation policies
- Error tracking and alerts

## Project Structure

```
trader_joe/
├── config/                     # Configuration files
│   ├── config.example.yaml    # Example configuration template
│   └── config.yaml           # Active configuration
├── data/                      # Data storage
│   └── trader_joe.db         # SQLite database
├── logs/                      # Application logs
│   └── trader_joe.log        # Main log file
├── src/                       # Source code
│   ├── analysis/             # Analysis modules
│   │   ├── deep_optimizer.py    # ML optimization
│   │   └── sentiment_analysis.py # Sentiment analysis
│   ├── data/                 # Data handling
│   │   ├── finnhub_client.py    # API client
│   │   ├── market_data.py       # Market data
│   │   ├── models.py            # Data models
│   │   ├── news_scraper.py      # News collection
│   │   ├── rate_limiter.py      # API rate limiting
│   │   └── ticker_mapper.py     # Symbol mapping
│   ├── engine/               # Trading logic
│   │   └── recommendation.py    # Trade recommendations
│   ├── utils/                # Utilities
│   │   └── text_processing.py   # Text processing
│   ├── web/                  # Web interface
│   │   ├── assets/             # Web components
│   │   ├── static/             # Static files
│   │   ├── templates/          # HTML templates
│   │   └── app.py              # Flask application
│   └── main.py              # Application entry
├── tests/                    # Test suite
│   ├── test_core.py         # Core tests
│   ├── test_market_data.py  # Market data tests
│   └── test_ticker_mapper.py # Ticker tests
└── requirements.txt         # Dependencies
