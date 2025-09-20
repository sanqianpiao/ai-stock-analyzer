#!/usr/bin/env python3
"""
Main CLI Interface for AI Stock Analyzer.

This script provides a command-line interface for analyzing stocks using
technical analysis, sentiment analysis, and AI-powered insights.
"""

import argparse
import sys
import logging
from datetime import datetime
from typing import Optional

# Add src directory to path for imports
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config import config
    from src.data_fetcher import DataFetcher, validate_ticker
    from src.analyzer import TechnicalAnalyzer, SentimentAnalyzer
    from src.analyzer import analyze_stock, analyze_news_sentiment
    from src.reporter import StockReporter
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)


class StockAnalyzerCLI:
    """Command-line interface for the AI Stock Analyzer."""
    
    def __init__(self):
        """Initialize the CLI with analyzer components."""
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.reporter = StockReporter()
    
    def validate_setup(self) -> bool:
        """Validate that the application is properly configured."""
        if not config.validate_api_keys():
            print("❌ Missing required API keys. Please check your .env file.")
            print("Required keys: OPENAI_API_KEY, NEWSAPI_KEY")
            return False
        
        print("✅ API keys validated successfully")
        return True
    
    def analyze_ticker(
        self, 
        ticker: str, 
        period: str = None,
        include_news: bool = True,
        max_articles: Optional[int] = None
    ) -> dict:
        """
        Perform comprehensive analysis of a stock ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period for analysis
            include_news (bool): Whether to include news sentiment analysis
            max_articles (int): Maximum number of news articles to analyze
            
        Returns:
            dict: Complete analysis results
        """
        print(f"\n🔍 Analyzing {ticker.upper()}...")
        
        # Validate ticker
        if not validate_ticker(ticker):
            print(f"❌ Invalid ticker symbol: {ticker}")
            return {}
        
        results = {
            "ticker": ticker.upper(),
            "analysis_timestamp": datetime.now().isoformat(),
            "period": period or config.DEFAULT_PERIOD
        }
        
        # Fetch stock data
        print("📈 Fetching stock data...")
        stock_data = self.data_fetcher.fetch_stock_data(ticker, period)
        if stock_data is None or stock_data.empty:
            print(f"❌ Could not fetch stock data for {ticker}")
            return results
        
        print(f"✅ Fetched {len(stock_data)} days of stock data")
        
        # Get stock info
        stock_info = self.data_fetcher.get_stock_info(ticker)
        if stock_info:
            results["company_info"] = stock_info
            print(f"📊 Company: {stock_info.get('longName', 'N/A')}")
        
        # Perform technical analysis
        print("⚙️ Performing technical analysis...")
        technical_analysis = analyze_stock(stock_data)
        if technical_analysis:
            results["technical_analysis"] = technical_analysis
            print(f"✅ Technical analysis complete - Trend: {technical_analysis.get('trend', 'Unknown')}")
        
        # Perform sentiment analysis if requested
        if include_news:
            print("📰 Fetching and analyzing news sentiment...")
            news_articles = self.data_fetcher.fetch_news(ticker, max_articles)
            
            if news_articles:
                print(f"✅ Found {len(news_articles)} news articles")
                sentiment_result = analyze_news_sentiment(news_articles, ticker)
                results["sentiment_analysis"] = sentiment_result
                results["news_articles"] = news_articles
                print(f"✅ Sentiment analysis complete - {sentiment_result.get('sentiment_label', 'Unknown')}")
            else:
                print("⚠️ No recent news articles found")
                results["sentiment_analysis"] = {
                    "sentiment_score": 0.0,
                    "sentiment_label": "Neutral",
                    "explanation": "No recent news articles available"
                }
        
        # Store stock data for potential report generation
        results["stock_data"] = stock_data
        
        return results
    
    def print_analysis_summary(self, results: dict):
        """Print a formatted summary of the analysis results."""
        if not results:
            return
        
        ticker = results.get("ticker", "Unknown")
        print(f"\n{'='*60}")
        print(f"📊 STOCK ANALYSIS SUMMARY FOR {ticker}")
        print(f"{'='*60}")
        
        # Company information
        company_info = results.get("company_info", {})
        if company_info:
            print(f"\n🏢 COMPANY INFORMATION")
            print(f"Name: {company_info.get('longName', 'N/A')}")
            print(f"Sector: {company_info.get('sector', 'N/A')}")
            print(f"Industry: {company_info.get('industry', 'N/A')}")
            print(f"Market Cap: ${company_info.get('marketCap', 0):,.0f}")
        
        # Technical analysis
        technical = results.get("technical_analysis", {})
        if technical:
            print(f"\n📈 TECHNICAL ANALYSIS")
            print(f"Current Price: ${technical.get('current_price', 0):.2f}")
            print(f"Price Change: {technical.get('price_change', 0):+.2f} ({technical.get('price_change_pct', 0):+.2f}%)")
            print(f"Trend: {technical.get('trend', 'Unknown')}")
            print(f"Volume: {technical.get('volume', 0):,}")
            
            indicators = technical.get('technical_indicators', {})
            if indicators:
                print(f"\n📊 Technical Indicators:")
                print(f"  RSI: {indicators.get('rsi', 'N/A'):.1f}")
                print(f"  5-day MA: ${indicators.get('ma_5', 0):.2f}")
                print(f"  20-day MA: ${indicators.get('ma_20', 0):.2f}")
                print(f"  Bollinger Band Position: {indicators.get('bb_position', 0):.2f}")
            
            levels = technical.get('support_resistance', {})
            if levels:
                print(f"\n🎯 Support/Resistance Levels:")
                print(f"  Support: ${levels.get('support', 0):.2f}")
                print(f"  Resistance: ${levels.get('resistance', 0):.2f}")
                print(f"  Pivot Point: ${levels.get('pivot_point', 0):.2f}")
            
            # Trading signals
            signals = technical.get('signals', {})
            if signals:
                print(f"\n🚨 Trading Signals:")
                buy_signals = signals.get('buy', [])
                sell_signals = signals.get('sell', [])
                neutral_signals = signals.get('neutral', [])
                
                if buy_signals:
                    print(f"  🟢 Buy Signals: {', '.join(buy_signals)}")
                if sell_signals:
                    print(f"  🔴 Sell Signals: {', '.join(sell_signals)}")
                if neutral_signals:
                    print(f"  🟡 Neutral Signals: {', '.join(neutral_signals)}")
        
        # Sentiment analysis
        sentiment = results.get("sentiment_analysis", {})
        if sentiment and sentiment.get('sentiment_score') is not None:
            print(f"\n📰 SENTIMENT ANALYSIS")
            score = sentiment.get('sentiment_score', 0)
            label = sentiment.get('sentiment_label', 'Neutral')
            confidence = sentiment.get('confidence', 0)
            
            # Add emoji based on sentiment
            emoji = "😊" if score > 0.3 else "😔" if score < -0.3 else "😐"
            print(f"Sentiment: {emoji} {label} (Score: {score:.2f}, Confidence: {confidence:.1%})")
            
            explanation = sentiment.get('explanation', '')
            if explanation:
                print(f"Analysis: {explanation}")
            
            key_factors = sentiment.get('key_factors', [])
            if key_factors:
                print(f"Key Factors: {', '.join(key_factors)}")
            
            articles_count = sentiment.get('articles_analyzed', 0)
            if articles_count > 0:
                print(f"Based on {articles_count} recent news articles")
        
        # Generate overall recommendation
        self._print_recommendation(technical, sentiment)
        
        print(f"\n{'='*60}")
        print(f"Analysis completed at {results.get('analysis_timestamp', datetime.now().isoformat())}")
        print(f"⚠️  This is not financial advice. Always do your own research.")
        print(f"{'='*60}")
    
    def generate_report(self, results: dict, format_type: str = "html", output_file: str = None) -> Optional[str]:
        """
        Generate a comprehensive report from analysis results.
        
        Args:
            results (dict): Analysis results from analyze_ticker
            format_type (str): Report format ('html' or 'markdown')
            output_file (str): Optional output file path
            
        Returns:
            str: Path to generated report file, or None if failed
        """
        if not results or "stock_data" not in results:
            print("❌ No analysis results or stock data available for report generation")
            return None
        
        try:
            print(f"📄 Generating {format_type.upper()} report...")
            
            # Extract stock data
            stock_data = results["stock_data"]
            
            # Generate report
            report_path = self.reporter.generate_report(
                analysis_results=results,
                stock_data=stock_data,
                format_type=format_type,
                output_file=output_file
            )
            
            print(f"✅ Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            print(f"❌ Failed to generate report: {e}")
            return None
    
    def _print_recommendation(self, technical: dict, sentiment: dict):
        """Generate and print an overall recommendation."""
        print(f"\n🎯 OVERALL RECOMMENDATION")
        
        if not technical and not sentiment:
            print("Insufficient data for recommendation")
            return
        
        # Simple recommendation logic
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
            
            # Adjust based on RSI
            rsi = technical.get('technical_indicators', {}).get('rsi', 50)
            if rsi > 70:
                tech_score -= 0.5  # Overbought
            elif rsi < 30:
                tech_score += 0.5  # Oversold
        
        # Sentiment scoring
        if sentiment:
            sent_raw = sentiment.get('sentiment_score', 0)
            confidence = sentiment.get('confidence', 0)
            sent_score = sent_raw * confidence  # Weight by confidence
        
        # Combined score
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
        
        print(f"Recommendation: {recommendation}")
        print(f"Technical Score: {tech_score:.1f}, Sentiment Score: {sent_score:.1f}")
        print(f"Combined Score: {total_score:.1f}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Stock Analyzer - Analyze stocks with technical analysis and sentiment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL                          # Analyze Apple stock
  python main.py GOOGL --period 3mo           # Analyze Google with 3-month data
  python main.py TSLA --no-news               # Analyze Tesla without news sentiment
  python main.py MSFT --max-articles 5        # Limit news analysis to 5 articles
  python main.py NVDA --verbose               # Detailed logging output
        """
    )
    
    parser.add_argument(
        "ticker",
        nargs="?",
        help="Stock ticker symbol (e.g., AAPL, GOOGL, TSLA)"
    )
    
    parser.add_argument(
        "-p", "--period",
        choices=config.get_valid_periods(),
        default=config.DEFAULT_PERIOD,
        help=f"Time period for analysis (default: {config.DEFAULT_PERIOD})"
    )
    
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Skip news sentiment analysis"
    )
    
    parser.add_argument(
        "--max-articles",
        type=int,
        default=config.MAX_NEWS_ARTICLES,
        help=f"Maximum number of news articles to analyze (default: {config.MAX_NEWS_ARTICLES})"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and exit"
    )
    
    parser.add_argument(
        "--generate-report",
        choices=["html", "markdown", "both"],
        help="Generate a comprehensive report in the specified format(s) after analysis"
    )
    
    parser.add_argument(
        "--report-output",
        type=str,
        help="Output file path for the generated report (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="AI Stock Analyzer 0.1.0"
    )
    
    return parser


def main():
    """Main entry point for the CLI application."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Initialize CLI
    cli = StockAnalyzerCLI()
    
    # Validate setup
    if not cli.validate_setup():
        sys.exit(1)
    
    # If validation only, exit here
    if args.validate_only:
        print("✅ Configuration validation successful")
        sys.exit(0)
    
    # Check if ticker is provided for analysis
    if not args.ticker:
        print("❌ Error: ticker argument is required for analysis")
        print("Use --help to see usage information or --validate-only to just validate setup")
        sys.exit(1)
    
    try:
        # Perform analysis
        results = cli.analyze_ticker(
            ticker=args.ticker,
            period=args.period,
            include_news=not args.no_news,
            max_articles=args.max_articles
        )
        
        if results:
            cli.print_analysis_summary(results)
            
            # Generate reports if requested
            if args.generate_report:
                if args.generate_report == "both":
                    # Generate both HTML and Markdown reports
                    html_path = cli.generate_report(results, "html", args.report_output)
                    md_path = cli.generate_report(results, "markdown", 
                                                args.report_output.replace('.html', '.md') if args.report_output and args.report_output.endswith('.html') else None)
                    
                    if html_path and md_path:
                        print(f"\n📊 Reports generated:")
                        print(f"  HTML: {html_path}")
                        print(f"  Markdown: {md_path}")
                else:
                    # Generate single report
                    report_path = cli.generate_report(results, args.generate_report, args.report_output)
                    if report_path:
                        print(f"\n📊 Report generated: {report_path}")
        else:
            print("❌ Analysis failed. Please check the ticker symbol and try again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ An unexpected error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()