"""
Layout configuration for the Dash dashboard with optimized update intervals.
"""

from dash import html, dcc

def create_layout(update_intervals: dict):
    """
    Create the Dash app layout with optimized update intervals.
    
    Args:
        update_intervals: Dictionary containing update intervals for different components
    """
    return html.Div([
        # Header with Theme Toggle
        html.Header([
            html.Div([
                html.Div([
                    html.H1("Trader Joe Dashboard"),
                    html.P("Real-time market insights and trading recommendations", 
                           className="text-light mt-2 mb-0")
                ], className="header-content"),
                html.Div([
                    html.Button(
                        "🌙",
                        id="theme-toggle",
                        className="theme-toggle-btn",
                        title="Toggle dark/light mode"
                    )
                ], className="header-controls")
            ], className="container d-flex justify-content-between align-items-center")
        ]),

        # Real-Time Ticker (5s update)
        html.Div([
            html.Div([
                html.Div(
                    id="ticker-tape",
                    className="ticker-tape",
                    children=[
                        html.Div(className="ticker-content")
                    ]
                )
            ], className="container-fluid px-0")
        ], className="ticker-container mb-4"),

        # Main content
        html.Main([
            # Search and Filter Bar
            html.Section([
                html.Div([
                    html.Div([
                        dcc.Input(
                            id="stock-search",
                            type="text",
                            placeholder="Search stocks by symbol or name...",
                            className="form-control search-input"
                        ),
                        html.Div([
                            dcc.Dropdown(
                                id="sector-filter",
                                placeholder="Sector",
                                className="filter-dropdown"
                            ),
                            dcc.Dropdown(
                                id="market-cap-filter",
                                placeholder="Market Cap",
                                className="filter-dropdown"
                            ),
                            dcc.RangeSlider(
                                id="price-range-filter",
                                className="price-range-slider"
                            )
                        ], className="filter-controls mt-3")
                    ], className="col-12")
                ], className="row")
            ], className="container mb-4"),

            # Watchlist and Portfolio Section
            html.Section([
                html.Div([
                    # Watchlist Panel
                    html.Div([
                        html.Div([
                            html.H3("Watchlist"),
                            html.Button(
                                "➕",
                                id="add-to-watchlist-btn",
                                className="action-btn",
                                title="Add to watchlist"
                            )
                        ], className="d-flex justify-content-between align-items-center mb-3"),
                        html.Div(
                            id="watchlist-content",
                            className="watchlist-container"
                        )
                    ], className="col-md-6 mb-4"),

                    # Portfolio Panel
                    html.Div([
                        html.Div([
                            html.H3("Portfolio"),
                            html.Div([
                                html.Span(
                                    id="portfolio-value",
                                    className="portfolio-value mr-3"
                                ),
                                html.Span(
                                    id="portfolio-change",
                                    className="portfolio-change"
                                )
                            ])
                        ], className="d-flex justify-content-between align-items-center mb-3"),
                        html.Div(
                            id="portfolio-content",
                            className="portfolio-container"
                        )
                    ], className="col-md-6 mb-4")
                ], className="row")
            ], className="container mb-4"),

            # Charts section with Technical Indicators
            html.Section([
                html.Div([
                    # Market Overview
                    html.Div([
                        html.Div([
                            html.H3("Market Sentiment Overview"),
                            html.Div([
                                dcc.Dropdown(
                                    id="timeframe-selector",
                                    options=[
                                        {"label": "1D", "value": "1D"},
                                        {"label": "1W", "value": "1W"},
                                        {"label": "1M", "value": "1M"},
                                        {"label": "3M", "value": "3M"},
                                        {"label": "1Y", "value": "1Y"},
                                    ],
                                    value="1D",
                                    className="timeframe-dropdown"
                                )
                            ])
                        ], className="d-flex justify-content-between align-items-baseline"),
                        html.Div([
                            dcc.Graph(id='sentiment-chart')
                        ], className="chart-container")
                    ], className="col-12 mb-4"),

                    # Advanced Stock Analysis
                    html.Div([
                        html.Div([
                            html.H3("Stock Price Analysis"),
                            html.Div([
                                dcc.Dropdown(
                                    id='technical-indicators',
                                    multi=True,
                                    placeholder="Add technical indicators...",
                                    options=[
                                        {"label": "Moving Average (MA)", "value": "MA"},
                                        {"label": "RSI", "value": "RSI"},
                                        {"label": "MACD", "value": "MACD"},
                                        {"label": "Bollinger Bands", "value": "BB"}
                                    ],
                                    className="mb-3"
                                )
                            ])
                        ], className="d-flex justify-content-between align-items-baseline"),
                        html.Div([
                            dcc.Dropdown(
                                id='ticker-selector',
                                placeholder="Select a stock...",
                                className="mb-3"
                            ),
                            dcc.Graph(id='price-chart')
                        ], className="chart-container")
                    ], className="col-12 mb-4")
                ], className="row")
            ], className="container mb-4"),

            # Trading Insights Section
            html.Section([
                html.Div([
                    # Trading Suggestions Panel
                    html.Div([
                        html.Div([
                            html.H3("Trading Suggestions"),
                            html.Div([
                                dcc.Dropdown(
                                    id="risk-level-filter",
                                    options=[
                                        {"label": "Low Risk", "value": "low"},
                                        {"label": "Medium Risk", "value": "medium"},
                                        {"label": "High Risk", "value": "high"}
                                    ],
                                    placeholder="Filter by risk level",
                                    className="risk-filter"
                                )
                            ])
                        ]),
                        html.Div(
                            id='suggestions-container',
                            className="suggestions-wrapper"
                        )
                    ], className="col-md-6 mb-4 mb-md-0"),

                    # News Feed Panel
                    html.Div([
                        html.Div([
                            html.H3("Latest News"),
                            html.Div([
                                dcc.Dropdown(
                                    id="news-filter",
                                    options=[
                                        {"label": "All News", "value": "all"},
                                        {"label": "Watchlist", "value": "watchlist"},
                                        {"label": "Portfolio", "value": "portfolio"}
                                    ],
                                    value="all",
                                    className="news-filter"
                                )
                            ])
                        ]),
                        html.Div(
                            id='news-feed',
                            className="news-wrapper"
                        )
                    ], className="col-md-6")
                ], className="row")
            ], className="container")
        ]),

        # Optimized update intervals for different components
        dcc.Interval(
            id='market-data-interval',
            interval=update_intervals.get('market_data', 5000),  # 5s default
            n_intervals=0
        ),
        dcc.Interval(
            id='watchlist-interval',
            interval=update_intervals.get('watchlist', 10000),  # 10s default
            n_intervals=0
        ),
        dcc.Interval(
            id='news-interval',
            interval=update_intervals.get('news', 30000),  # 30s default
            n_intervals=0
        ),
        dcc.Interval(
            id='suggestions-interval',
            interval=update_intervals.get('suggestions', 60000),  # 60s default
            n_intervals=0
        ),

        # Store components for state management and caching
        dcc.Store(id='watchlist-store'),
        dcc.Store(id='portfolio-store'),
        dcc.Store(id='alerts-store'),
        dcc.Store(id='preferences-store'),
        dcc.Store(id='market-data-cache'),  # Cache for market data
        dcc.Store(id='news-cache'),  # Cache for news data
        dcc.Store(id='suggestions-cache'),  # Cache for trading suggestions

        # Modals
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("Add to Watchlist", id="add-stock-modal-title"),
                        html.Button(
                            "×",
                            id="close-add-stock-modal",
                            className="close",
                            **{"data-dismiss": "modal"}
                        )
                    ], className="modal-header"),
                    html.Div([
                        dcc.Input(
                            id="add-stock-symbol",
                            type="text",
                            placeholder="Enter stock symbol (e.g., AAPL)",
                            className="form-control mb-3"
                        ),
                        dcc.Input(
                            id="add-stock-notes",
                            type="text",
                            placeholder="Notes (optional)",
                            className="form-control mb-3"
                        ),
                        html.Div([
                            dcc.Input(
                                id="add-stock-quantity",
                                type="number",
                                placeholder="Quantity",
                                className="form-control mb-3"
                            ),
                            dcc.Input(
                                id="add-stock-price",
                                type="number",
                                placeholder="Average Cost",
                                className="form-control mb-3"
                            )
                        ], id="portfolio-fields", style={'display': 'none'})
                    ], className="modal-body"),
                    html.Div([
                        html.Button(
                            "Add",
                            id="confirm-add-stock",
                            className="btn btn-primary"
                        )
                    ], className="modal-footer")
                ], className="modal-content")
            ], className="modal-dialog")
        ], id="add-stock-modal", className="modal fade", role="dialog"),

        # Loading states
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
        dcc.Loading(
            id="loading-2",
            type="default",
            children=html.Div(id="loading-output-2")
        )
    ], className="dashboard-container", id="dashboard-container")
