"""
Enhanced dashboard component for market data visualization.
"""
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_market_overview(app):
    """Create an enhanced market overview component."""
    
    @app.callback(
        Output('market-data-graph', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_market_data(_):
        try:
            # Placeholder for market data fetching logic
            dates = [datetime.now() - timedelta(minutes=x) for x in range(60, 0, -1)]
            prices = [100 + x * 0.1 for x in range(60)]  # Sample data
            volumes = [1000 + x * 10 for x in range(60)]  # Sample data

            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                name='Price',
                line=dict(color='#8884d8'),
                yaxis='y'
            ))
            
            # Add volume bars
            fig.add_trace(go.Bar(
                x=dates,
                y=volumes,
                name='Volume',
                marker=dict(color='#82ca9d'),
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title='Market Overview',
                xaxis=dict(title='Time'),
                yaxis=dict(
                    title='Price',
                    titlefont=dict(color='#8884d8'),
                    tickfont=dict(color='#8884d8')
                ),
                yaxis2=dict(
                    title='Volume',
                    titlefont=dict(color='#82ca9d'),
                    tickfont=dict(color='#82ca9d'),
                    overlaying='y',
                    side='right'
                ),
                showlegend=True,
                height=400,
                template='plotly_dark'
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure with error message
            return go.Figure().add_annotation(
                text=f"Error loading market data: {str(e)}",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14)
            )

    return html.Div(
        className="market-overview",
        children=[
            html.H3("Enhanced Market Overview"),
            dcc.Graph(
                id='market-data-graph',
                config={'displayModeBar': False}
            )
        ]
    )
