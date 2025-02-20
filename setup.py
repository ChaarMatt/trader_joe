from setuptools import setup, find_packages

setup(
    name="trader_joe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'flask>=2.0.0',
        'flask-socketio>=5.0.0',
        'dash>=2.0.0',
        'plotly>=5.0.0',
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'yfinance>=0.1.70',
        'newsapi-python>=0.2.6',
        'aiohttp>=3.8.0',
        'beautifulsoup4>=4.9.3',
        'html5lib>=1.1',
        'spacy>=3.0.0',
        'textblob>=0.15.3',
        'nltk>=3.6.0',
        'SQLAlchemy>=1.4.0',
        'python-dotenv>=0.19.0',
        'pyyaml>=5.4.0',
        'python-dateutil>=2.8.2'
    ],
)
