"""
Report Generation Module for AI Stock Analyzer.

This module provides functionality to generate comprehensive reports in markdown and HTML format,
including interactive charts, tables, and proper citations for news sources.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import base64
from io import BytesIO

from .config import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class StockReporter:
    """Class for generating comprehensive stock analysis reports."""
    
    def __init__(self):
        """Initialize the StockReporter."""
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        
        # Set plotly theme
        pio.templates.default = "plotly_white"
    
    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        stock_data: pd.DataFrame,
        format_type: str = "html",
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive stock analysis report.
        
        Args:
            analysis_results (dict): Complete analysis results from CLI
            stock_data (pd.DataFrame): Stock price data
            format_type (str): Output format ('html', 'markdown')
            output_file (str, optional): Output file path
            
        Returns:
            str: Path to generated report file
        """
        ticker = analysis_results.get("ticker", "UNKNOWN")
        timestamp = analysis_results.get("analysis_timestamp", datetime.now().isoformat())
        
        if not output_file:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "html" if format_type == "html" else "md"
            output_file = self.report_dir / f"{ticker}_analysis_{timestamp_str}.{extension}"
        
        try:
            if format_type == "html":
                content = self._generate_html_report(analysis_results, stock_data)
            else:
                content = self._generate_markdown_report(analysis_results, stock_data)
            
            # Write report to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated {format_type.upper()} report: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _generate_html_report(self, results: Dict[str, Any], stock_data: pd.DataFrame) -> str:
        """Generate HTML report with interactive charts."""
        ticker = results.get("ticker", "UNKNOWN")
        timestamp = results.get("analysis_timestamp", datetime.now().isoformat())
        
        # Generate charts
        price_chart = self._create_price_chart(stock_data, results)
        technical_chart = self._create_technical_indicators_chart(stock_data, results)
        sentiment_chart = self._create_sentiment_chart(results)
        
        # Convert charts to HTML
        price_chart_html = price_chart.to_html(include_plotlyjs='cdn', div_id="price-chart")
        technical_chart_html = technical_chart.to_html(include_plotlyjs=False, div_id="technical-chart")
        sentiment_chart_html = sentiment_chart.to_html(include_plotlyjs=False, div_id="sentiment-chart")
        
        # Generate report content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Stock Analysis Report</title>
    <style>
        {self._get_html_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>📊 {ticker} Stock Analysis Report</h1>
            <div class="timestamp">Generated on {datetime.fromisoformat(timestamp).strftime('%B %d, %Y at %I:%M %p')}</div>
        </header>
        
        <div class="summary-cards">
            {self._generate_summary_cards(results)}
        </div>
        
        <section class="company-info">
            <h2>🏢 Company Information</h2>
            {self._generate_company_info_html(results)}
        </section>
        
        <section class="price-analysis">
            <h2>📈 Price Analysis</h2>
            {price_chart_html}
        </section>
        
        <section class="technical-analysis">
            <h2>⚙️ Technical Analysis</h2>
            {self._generate_technical_analysis_html(results)}
            {technical_chart_html}
        </section>
        
        <section class="sentiment-analysis">
            <h2>📰 Sentiment Analysis</h2>
            {self._generate_sentiment_analysis_html(results)}
            {sentiment_chart_html}
        </section>
        
        <section class="news-sources">
            <h2>📑 News Sources & Citations</h2>
            {self._generate_news_citations_html(results)}
        </section>
        
        <section class="recommendation">
            <h2>🎯 Investment Recommendation</h2>
            {self._generate_recommendation_html(results)}
        </section>
        
        <footer class="report-footer">
            <div class="disclaimer">
                <p><strong>⚠️ Disclaimer:</strong> This analysis is for informational purposes only and should not be considered as financial advice. 
                Always conduct your own research and consult with qualified financial advisors before making investment decisions.</p>
            </div>
            <div class="generation-info">
                <p>Report generated by AI Stock Analyzer v0.1.0 | Powered by OpenAI GPT & yfinance</p>
            </div>
        </footer>
    </div>
</body>
</html>
"""
        return html_content
    
    def _generate_markdown_report(self, results: Dict[str, Any], stock_data: pd.DataFrame) -> str:
        """Generate Markdown report."""
        ticker = results.get("ticker", "UNKNOWN")
        timestamp = results.get("analysis_timestamp", datetime.now().isoformat())
        
        # Generate charts and save as images
        price_chart = self._create_price_chart(stock_data, results)
        technical_chart = self._create_technical_indicators_chart(stock_data, results)
        
        # Save charts as images for markdown
        price_img_path = self.report_dir / f"{ticker}_price_chart.png"
        technical_img_path = self.report_dir / f"{ticker}_technical_chart.png"
        
        price_chart.write_image(str(price_img_path), width=800, height=500)
        technical_chart.write_image(str(technical_img_path), width=800, height=400)
        
        markdown_content = f"""# 📊 {ticker} Stock Analysis Report

**Generated on:** {datetime.fromisoformat(timestamp).strftime('%B %d, %Y at %I:%M %p')}

---

## 📋 Executive Summary

{self._generate_executive_summary(results)}

---

## 🏢 Company Information

{self._generate_company_info_markdown(results)}

---

## 📈 Price Analysis

![Price Chart]({price_img_path.name})

{self._generate_price_analysis_markdown(results, stock_data)}

---

## ⚙️ Technical Analysis

![Technical Indicators]({technical_img_path.name})

{self._generate_technical_analysis_markdown(results)}

---

## 📰 Sentiment Analysis

{self._generate_sentiment_analysis_markdown(results)}

---

## 📑 News Sources & Citations

{self._generate_news_citations_markdown(results)}

---

## 🎯 Investment Recommendation

{self._generate_recommendation_markdown(results)}

---

## ⚠️ Disclaimer

This analysis is for informational purposes only and should not be considered as financial advice. 
Always conduct your own research and consult with qualified financial advisors before making investment decisions.

---

*Report generated by AI Stock Analyzer v0.1.0 | Powered by OpenAI GPT & yfinance*
"""
        return markdown_content
    
    def _create_price_chart(self, stock_data: pd.DataFrame, results: Dict[str, Any]) -> go.Figure:
        """Create interactive price chart with technical indicators."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Stock Price & Moving Averages', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Moving averages if available
        if 'MA_5' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['MA_5'],
                    mode='lines',
                    name='5-day MA',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        if 'MA_20' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['MA_20'],
                    mode='lines',
                    name='20-day MA',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
        
        # Volume chart
        colors = ['green' if row['Close'] > row['Open'] else 'red' for _, row in stock_data.iterrows()]
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        ticker = results.get("ticker", "STOCK")
        fig.update_layout(
            title=f'{ticker} Stock Price Analysis',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    def _create_technical_indicators_chart(self, stock_data: pd.DataFrame, results: Dict[str, Any]) -> go.Figure:
        """Create technical indicators chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('RSI', 'Bollinger Bands'),
            row_heights=[0.3, 0.7]
        )
        
        # RSI chart
        if 'RSI' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=1, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
        
        # Bollinger Bands
        if all(col in stock_data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='red', width=1),
                    fill=None
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='red', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=2)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Technical Indicators',
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="RSI", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        
        return fig
    
    def _create_sentiment_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create sentiment analysis visualization."""
        sentiment = results.get("sentiment_analysis", {})
        
        # Sentiment gauge chart
        score = sentiment.get("sentiment_score", 0)
        confidence = sentiment.get("confidence", 0)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Sentiment Score (Confidence: {confidence:.1%})"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightcoral"},
                    {'range': [-0.3, 0.3], 'color': "lightyellow"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        
        fig.update_layout(
            title="News Sentiment Analysis",
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .report-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        
        .report-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .timestamp {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        
        .card .change {
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .neutral { color: #6c757d; }
        
        section {
            margin-bottom: 40px;
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .info-item {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .info-item strong {
            color: #667eea;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .metrics-table th {
            background-color: #667eea;
            color: white;
        }
        
        .metrics-table tr:hover {
            background-color: #f5f5f5;
        }
        
        .news-item {
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background: #f8f9fa;
        }
        
        .news-item h4 {
            color: #333;
            margin-bottom: 8px;
        }
        
        .news-item .source {
            color: #667eea;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .news-item .date {
            color: #6c757d;
            font-size: 0.85em;
        }
        
        .recommendation-box {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2em;
        }
        
        .recommendation-box.sell {
            background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        }
        
        .recommendation-box.hold {
            background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        }
        
        .report-footer {
            margin-top: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: center;
        }
        
        .disclaimer {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .summary-cards {
                grid-template-columns: 1fr;
            }
            
            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _generate_summary_cards(self, results: Dict[str, Any]) -> str:
        """Generate summary cards for HTML report."""
        technical = results.get("technical_analysis", {})
        sentiment = results.get("sentiment_analysis", {})
        
        current_price = technical.get("current_price", 0)
        price_change = technical.get("price_change", 0)
        price_change_pct = technical.get("price_change_pct", 0)
        trend = technical.get("trend", "Unknown")
        sentiment_label = sentiment.get("sentiment_label", "Neutral")
        sentiment_score = sentiment.get("sentiment_score", 0)
        
        change_class = "positive" if price_change >= 0 else "negative"
        sentiment_class = "positive" if sentiment_score > 0.3 else "negative" if sentiment_score < -0.3 else "neutral"
        
        return f"""
        <div class="card">
            <h3>💰 Current Price</h3>
            <div class="value">${current_price:.2f}</div>
            <div class="change {change_class}">
                {'+' if price_change >= 0 else ''}{price_change:.2f} ({'+' if price_change_pct >= 0 else ''}{price_change_pct:.2f}%)
            </div>
        </div>
        <div class="card">
            <h3>📈 Trend</h3>
            <div class="value">{trend}</div>
        </div>
        <div class="card">
            <h3>📰 Sentiment</h3>
            <div class="value {sentiment_class}">{sentiment_label}</div>
            <div class="change">Score: {sentiment_score:.2f}</div>
        </div>
        """
    
    def _generate_company_info_html(self, results: Dict[str, Any]) -> str:
        """Generate company information HTML."""
        company_info = results.get("company_info", {})
        
        if not company_info:
            return "<p>Company information not available.</p>"
        
        return f"""
        <div class="info-grid">
            <div class="info-item">
                <strong>Company:</strong> {company_info.get('longName', 'N/A')}
            </div>
            <div class="info-item">
                <strong>Sector:</strong> {company_info.get('sector', 'N/A')}
            </div>
            <div class="info-item">
                <strong>Industry:</strong> {company_info.get('industry', 'N/A')}
            </div>
            <div class="info-item">
                <strong>Market Cap:</strong> ${company_info.get('marketCap', 0):,.0f}
            </div>
            <div class="info-item">
                <strong>Beta:</strong> {company_info.get('beta', 'N/A')}
            </div>
            <div class="info-item">
                <strong>P/E Ratio:</strong> {company_info.get('trailingPE', 'N/A')}
            </div>
        </div>
        """
    
    def _generate_technical_analysis_html(self, results: Dict[str, Any]) -> str:
        """Generate technical analysis HTML."""
        technical = results.get("technical_analysis", {})
        indicators = technical.get("technical_indicators", {})
        levels = technical.get("support_resistance", {})
        signals = technical.get("signals", {})
        
        return f"""
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Indicator</th>
                    <th>Value</th>
                    <th>Signal</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>RSI (14-day)</td>
                    <td>{indicators.get('rsi', 'N/A')}</td>
                    <td>{'Overbought' if indicators.get('rsi', 50) > 70 else 'Oversold' if indicators.get('rsi', 50) < 30 else 'Neutral'}</td>
                </tr>
                <tr>
                    <td>5-day MA</td>
                    <td>${indicators.get('ma_5', 0):.2f}</td>
                    <td>{'Above' if technical.get('current_price', 0) > indicators.get('ma_5', 0) else 'Below'} current price</td>
                </tr>
                <tr>
                    <td>20-day MA</td>
                    <td>${indicators.get('ma_20', 0):.2f}</td>
                    <td>{'Above' if technical.get('current_price', 0) > indicators.get('ma_20', 0) else 'Below'} current price</td>
                </tr>
                <tr>
                    <td>Support Level</td>
                    <td>${levels.get('support', 0):.2f}</td>
                    <td>Key support</td>
                </tr>
                <tr>
                    <td>Resistance Level</td>
                    <td>${levels.get('resistance', 0):.2f}</td>
                    <td>Key resistance</td>
                </tr>
            </tbody>
        </table>
        
        <h3>Trading Signals</h3>
        <div class="info-grid">
            {f'<div class="info-item positive"><strong>Buy Signals:</strong> {", ".join(signals.get("buy", ["None"]))}</div>' if signals.get("buy") else ''}
            {f'<div class="info-item negative"><strong>Sell Signals:</strong> {", ".join(signals.get("sell", ["None"]))}</div>' if signals.get("sell") else ''}
            {f'<div class="info-item neutral"><strong>Neutral Signals:</strong> {", ".join(signals.get("neutral", ["None"]))}</div>' if signals.get("neutral") else ''}
        </div>
        """
    
    def _generate_sentiment_analysis_html(self, results: Dict[str, Any]) -> str:
        """Generate sentiment analysis HTML."""
        sentiment = results.get("sentiment_analysis", {})
        
        if not sentiment:
            return "<p>Sentiment analysis not available.</p>"
        
        score = sentiment.get("sentiment_score", 0)
        label = sentiment.get("sentiment_label", "Neutral")
        confidence = sentiment.get("confidence", 0)
        explanation = sentiment.get("explanation", "")
        key_factors = sentiment.get("key_factors", [])
        articles_count = sentiment.get("articles_analyzed", 0)
        
        sentiment_class = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"
        
        return f"""
        <div class="info-grid">
            <div class="info-item {sentiment_class}">
                <strong>Sentiment:</strong> {label}
            </div>
            <div class="info-item">
                <strong>Score:</strong> {score:.2f} (-1.0 to 1.0)
            </div>
            <div class="info-item">
                <strong>Confidence:</strong> {confidence:.1%}
            </div>
            <div class="info-item">
                <strong>Articles Analyzed:</strong> {articles_count}
            </div>
        </div>
        
        <h3>Analysis</h3>
        <p>{explanation}</p>
        
        {f'<h3>Key Factors</h3><ul>{"".join([f"<li>{factor}</li>" for factor in key_factors])}</ul>' if key_factors else ''}
        """
    
    def _generate_news_citations_html(self, results: Dict[str, Any]) -> str:
        """Generate news sources citations HTML."""
        news_articles = results.get("news_articles", [])
        
        if not news_articles:
            return "<p>No news articles were analyzed.</p>"
        
        citations_html = ""
        for i, article in enumerate(news_articles, 1):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown source')
            url = article.get('url', '#')
            published_at = article.get('published_at', '')
            
            # Format date
            try:
                if published_at:
                    date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%B %d, %Y')
                else:
                    formatted_date = 'Date unknown'
            except:
                formatted_date = 'Date unknown'
            
            citations_html += f"""
            <div class="news-item">
                <h4><a href="{url}" target="_blank">{title}</a></h4>
                <div class="source">Source: {source}</div>
                <div class="date">Published: {formatted_date}</div>
            </div>
            """
        
        return citations_html
    
    def _generate_recommendation_html(self, results: Dict[str, Any]) -> str:
        """Generate investment recommendation HTML."""
        # This would contain the same recommendation logic as in the CLI
        technical = results.get("technical_analysis", {})
        sentiment = results.get("sentiment_analysis", {})
        
        # Simple recommendation logic (same as CLI)
        tech_score = 0
        sent_score = 0
        
        # Technical scoring
        if technical:
            trend = technical.get('trend', '')
            if 'Strong Bullish' in trend:
                tech_score = 2
            elif 'Bullish' in trend:
                tech_score = 1
            elif 'Strong Bearish' in trend:
                tech_score = -2
            elif 'Bearish' in trend:
                tech_score = -1
        
        # Sentiment scoring
        if sentiment:
            sent_raw = sentiment.get('sentiment_score', 0)
            confidence = sentiment.get('confidence', 0)
            sent_score = sent_raw * confidence
        
        # Combined score
        total_score = tech_score + sent_score
        
        if total_score > 1.5:
            recommendation = "STRONG BUY"
            rec_class = "buy"
        elif total_score > 0.5:
            recommendation = "BUY"
            rec_class = "buy"
        elif total_score > -0.5:
            recommendation = "HOLD"
            rec_class = "hold"
        elif total_score > -1.5:
            recommendation = "SELL"
            rec_class = "sell"
        else:
            recommendation = "STRONG SELL"
            rec_class = "sell"
        
        return f"""
        <div class="recommendation-box {rec_class}">
            <h3>Investment Recommendation: {recommendation}</h3>
            <p>Technical Score: {tech_score:.1f} | Sentiment Score: {sent_score:.1f} | Combined Score: {total_score:.1f}</p>
        </div>
        """
    
    # Additional methods for markdown generation would go here...
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary for markdown."""
        technical = results.get("technical_analysis", {})
        sentiment = results.get("sentiment_analysis", {})
        
        current_price = technical.get("current_price", 0)
        trend = technical.get("trend", "Unknown")
        sentiment_label = sentiment.get("sentiment_label", "Neutral")
        
        return f"""
**Current Price:** ${current_price:.2f}  
**Trend:** {trend}  
**Sentiment:** {sentiment_label}  
**Analysis Period:** {results.get("period", "1mo")}
"""
    
    def _generate_company_info_markdown(self, results: Dict[str, Any]) -> str:
        """Generate company info for markdown."""
        company_info = results.get("company_info", {})
        
        if not company_info:
            return "Company information not available."
        
        return f"""
| Field | Value |
|-------|-------|
| Company | {company_info.get('longName', 'N/A')} |
| Sector | {company_info.get('sector', 'N/A')} |
| Industry | {company_info.get('industry', 'N/A')} |
| Market Cap | ${company_info.get('marketCap', 0):,.0f} |
| Beta | {company_info.get('beta', 'N/A')} |
| P/E Ratio | {company_info.get('trailingPE', 'N/A')} |
"""
    
    def _generate_price_analysis_markdown(self, results: Dict[str, Any], stock_data: pd.DataFrame) -> str:
        """Generate price analysis for markdown."""
        technical = results.get("technical_analysis", {})
        
        return f"""
**Current Price:** ${technical.get('current_price', 0):.2f}  
**Price Change:** {technical.get('price_change', 0):+.2f} ({technical.get('price_change_pct', 0):+.2f}%)  
**Trend:** {technical.get('trend', 'Unknown')}  
**Volume:** {technical.get('volume', 0):,}  

The chart above shows the stock's price movement with moving averages and volume analysis.
"""
    
    def _generate_technical_analysis_markdown(self, results: Dict[str, Any]) -> str:
        """Generate technical analysis for markdown."""
        technical = results.get("technical_analysis", {})
        indicators = technical.get("technical_indicators", {})
        levels = technical.get("support_resistance", {})
        
        return f"""
| Indicator | Value | Signal |
|-----------|-------|--------|
| RSI (14-day) | {indicators.get('rsi', 'N/A')} | {'Overbought' if indicators.get('rsi', 50) > 70 else 'Oversold' if indicators.get('rsi', 50) < 30 else 'Neutral'} |
| 5-day MA | ${indicators.get('ma_5', 0):.2f} | {'Above' if technical.get('current_price', 0) > indicators.get('ma_5', 0) else 'Below'} current price |
| 20-day MA | ${indicators.get('ma_20', 0):.2f} | {'Above' if technical.get('current_price', 0) > indicators.get('ma_20', 0) else 'Below'} current price |
| Support Level | ${levels.get('support', 0):.2f} | Key support |
| Resistance Level | ${levels.get('resistance', 0):.2f} | Key resistance |
"""
    
    def _generate_sentiment_analysis_markdown(self, results: Dict[str, Any]) -> str:
        """Generate sentiment analysis for markdown."""
        sentiment = results.get("sentiment_analysis", {})
        
        if not sentiment:
            return "Sentiment analysis not available."
        
        score = sentiment.get("sentiment_score", 0)
        label = sentiment.get("sentiment_label", "Neutral")
        confidence = sentiment.get("confidence", 0)
        explanation = sentiment.get("explanation", "")
        key_factors = sentiment.get("key_factors", [])
        articles_count = sentiment.get("articles_analyzed", 0)
        
        factors_text = ""
        if key_factors:
            factors_text = "**Key Factors:**\n" + "\n".join([f"- {factor}" for factor in key_factors])
        
        return f"""
**Sentiment:** {label}  
**Score:** {score:.2f} (-1.0 to 1.0)  
**Confidence:** {confidence:.1%}  
**Articles Analyzed:** {articles_count}  

**Analysis:** {explanation}

{factors_text}
"""
    
    def _generate_news_citations_markdown(self, results: Dict[str, Any]) -> str:
        """Generate news citations for markdown."""
        news_articles = results.get("news_articles", [])
        
        if not news_articles:
            return "No news articles were analyzed."
        
        citations_md = ""
        for i, article in enumerate(news_articles, 1):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown source')
            url = article.get('url', '#')
            published_at = article.get('published_at', '')
            
            # Format date
            try:
                if published_at:
                    date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%B %d, %Y')
                else:
                    formatted_date = 'Date unknown'
            except:
                formatted_date = 'Date unknown'
            
            citations_md += f"""
### {i}. [{title}]({url})
**Source:** {source}  
**Published:** {formatted_date}

"""
        
        return citations_md
    
    def _generate_recommendation_markdown(self, results: Dict[str, Any]) -> str:
        """Generate recommendation for markdown."""
        # Same logic as HTML version
        technical = results.get("technical_analysis", {})
        sentiment = results.get("sentiment_analysis", {})
        
        tech_score = 0
        sent_score = 0
        
        if technical:
            trend = technical.get('trend', '')
            if 'Strong Bullish' in trend:
                tech_score = 2
            elif 'Bullish' in trend:
                tech_score = 1
            elif 'Strong Bearish' in trend:
                tech_score = -2
            elif 'Bearish' in trend:
                tech_score = -1
        
        if sentiment:
            sent_raw = sentiment.get('sentiment_score', 0)
            confidence = sentiment.get('confidence', 0)
            sent_score = sent_raw * confidence
        
        total_score = tech_score + sent_score
        
        if total_score > 1.5:
            recommendation = "🟢 STRONG BUY"
        elif total_score > 0.5:
            recommendation = "🟢 BUY"
        elif total_score > -0.5:
            recommendation = "🟡 HOLD"
        elif total_score > -1.5:
            recommendation = "🔴 SELL"
        else:
            recommendation = "🔴 STRONG SELL"
        
        return f"""
## {recommendation}

**Technical Score:** {tech_score:.1f}  
**Sentiment Score:** {sent_score:.1f}  
**Combined Score:** {total_score:.1f}

Based on the technical analysis and sentiment analysis, this is the recommended action for the stock.
"""


# Create global reporter instance
stock_reporter = StockReporter()


# Convenience functions
def generate_report(analysis_results: Dict[str, Any], stock_data: pd.DataFrame, format_type: str = "html") -> str:
    """Convenience function to generate a report."""
    return stock_reporter.generate_report(analysis_results, stock_data, format_type)