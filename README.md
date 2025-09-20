# 🤖 AI Stock Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI%20GPT-orange.svg)](https://openai.com)

A powerful Python-based tool that leverages **Generative AI** to analyze stocks through technical analysis, sentiment analysis, and AI-powered insights. This project demonstrates practical applications of AI agents, RAG principles, and prompt engineering in financial analysis.

## ✨ Features

### 🔍 **Comprehensive Stock Analysis**
- **Technical Analysis**: RSI, Moving Averages, Bollinger Bands, Volume indicators
- **Sentiment Analysis**: AI-powered news sentiment using OpenAI GPT models
- **Trend Detection**: Bullish, Bearish, and Sideways trend identification
- **Support/Resistance Levels**: Automated level calculation
- **Trading Signals**: Buy/Sell/Hold recommendations

### 🤖 **AI-Powered Insights**
- **OpenAI Integration**: GPT-4o-mini for cost-effective sentiment analysis
- **Structured Prompts**: Consistent, reliable AI outputs
- **Confidence Scoring**: AI confidence levels for sentiment analysis
- **Error Handling**: Graceful fallbacks when APIs are unavailable

### 📊 **Data Sources**
- **Stock Data**: Real-time and historical data via yfinance
- **News Data**: Recent news articles via NewsAPI
- **Technical Indicators**: Professional-grade calculations

### 💻 **User-Friendly Interface**
- **Command Line Interface**: Simple, powerful CLI with argparse
- **Rich Output**: Emoji-enhanced, formatted analysis reports
- **Flexible Options**: Customizable analysis periods and parameters
- **Verbose Logging**: Detailed debugging information

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- NewsAPI key ([Get one here](https://newsapi.org/register))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-stock-analyzer.git
   cd ai-stock-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # OPENAI_API_KEY=your_openai_key_here
   # NEWSAPI_KEY=your_newsapi_key_here
   ```

4. **Validate setup**
   ```bash
   python main.py --validate-only
   ```

### Usage Examples

```bash
# Basic analysis
python main.py AAPL

# Extended period analysis
python main.py GOOGL --period 3mo

# Technical analysis only (no news)
python main.py TSLA --no-news

# Limit news articles
python main.py MSFT --max-articles 5

# Verbose output for debugging
python main.py NVDA --verbose
```

## 📋 Sample Output

```
🔍 Analyzing AAPL...
📈 Fetching stock data...
✅ Fetched 30 days of stock data
📊 Company: Apple Inc.
⚙️ Performing technical analysis...
✅ Technical analysis complete - Trend: Bullish
📰 Fetching and analyzing news sentiment...
✅ Found 8 news articles
✅ Sentiment analysis complete - Positive

============================================================
📊 STOCK ANALYSIS SUMMARY FOR AAPL
============================================================

🏢 COMPANY INFORMATION
Name: Apple Inc.
Sector: Technology
Industry: Consumer Electronics
Market Cap: $3,400,000,000,000

📈 TECHNICAL ANALYSIS
Current Price: $220.85
Price Change: +2.73 (+1.25%)
Trend: Bullish
Volume: 52,722,800

📊 Technical Indicators:
  RSI: 65.3
  5-day MA: $218.20
  20-day MA: $215.40
  Bollinger Band Position: 0.75

🎯 Support/Resistance Levels:
  Support: $216.10
  Resistance: $226.05
  Pivot Point: $221.20

🚨 Trading Signals:
  🟢 Buy Signals: Price above both MAs - uptrend
  🟡 Neutral Signals: RSI neutral (65.3)

📰 SENTIMENT ANALYSIS
Sentiment: 😊 Positive (Score: 0.67, Confidence: 82.0%)
Analysis: Recent product launches and strong quarterly outlook drive positive sentiment
Key Factors: Product innovation, Strong financials, Market leadership
Based on 8 recent news articles

🎯 OVERALL RECOMMENDATION
Recommendation: 🟢 BUY
Technical Score: 1.0, Sentiment Score: 0.5
Combined Score: 1.5

============================================================
Analysis completed at 2024-09-20T15:30:45.123456
⚠️  This is not financial advice. Always do your own research.
============================================================
```

## 🏗️ Project Structure

```
ai_stock_analyzer/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example              # Template for API keys
├── .gitignore               # Git ignore rules
├── main.py                  # CLI entry point
├── pytest.ini              # Test configuration
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── data_fetcher.py      # Stock and news data fetching
│   └── analyzer.py          # Technical and sentiment analysis
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   └── test_data_fetcher.py
├── mocks/                   # Mock data for testing
│   └── sample_stock_data.json
└── documents/
    └── specs.md             # Project specifications
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for sentiment analysis | Yes | - |
| `NEWSAPI_KEY` | NewsAPI key for news fetching | Yes | - |
| `OPENAI_MODEL` | OpenAI model to use | No | `gpt-4o-mini` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `DEFAULT_PERIOD` | Default analysis period | No | `1mo` |
| `MAX_NEWS_ARTICLES` | Max news articles to analyze | No | `10` |

### Technical Analysis Parameters

- **RSI Period**: 14 days (industry standard)
- **Moving Averages**: 5-day and 20-day
- **Bollinger Bands**: 20-day period, 2 standard deviations
- **Volume Analysis**: 20-day volume moving average

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_fetcher.py

# Run with verbose output
pytest -v
```

## 🚦 CLI Options

| Option | Description | Example |
|--------|-------------|---------|
| `ticker` | Stock ticker symbol (required) | `AAPL` |
| `-p, --period` | Analysis period | `--period 3mo` |
| `--no-news` | Skip news sentiment analysis | `--no-news` |
| `--max-articles` | Limit news articles | `--max-articles 5` |
| `-v, --verbose` | Enable verbose logging | `--verbose` |
| `--validate-only` | Only validate setup | `--validate-only` |

### Valid Periods
`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

## 🤖 AI Features Demonstrated

### 1. **Sentiment Analysis with GPT**
- Structured prompts for consistent analysis
- JSON response parsing with fallbacks
- Confidence scoring and key factor extraction

### 2. **Error Handling & Fallbacks**
- Graceful degradation when APIs are unavailable
- Mock data for testing and development
- Comprehensive error logging

### 3. **Prompt Engineering**
- Financial domain-specific prompts
- Temperature control for consistent outputs
- Response validation and sanitization

## 🔒 Security & Privacy

- **API Keys**: Stored in environment variables, never hardcoded
- **No Data Storage**: No sensitive financial data is stored locally
- **Rate Limiting**: Respects API rate limits and includes caching
- **Error Isolation**: API failures don't crash the application

## 🎯 Roadmap & Future Enhancements

### Phase 2: Advanced AI Features
- [ ] **RAG Implementation**: FAISS vector store for news retrieval
- [ ] **LangChain Agent**: ReAct-style agent orchestration
- [ ] **Multi-stock Comparison**: Portfolio-level analysis
- [ ] **Voice Interface**: Integration with speech-to-text

### Phase 3: Visualization & Deployment
- [ ] **Streamlit Web App**: Interactive dashboard
- [ ] **Plotly Charts**: Candlestick and technical indicator plots
- [ ] **Report Generation**: PDF/HTML report exports
- [ ] **Cloud Deployment**: Heroku/AWS deployment

### Phase 4: Advanced Analytics
- [ ] **Machine Learning Models**: Price prediction models
- [ ] **Risk Assessment**: VaR and portfolio risk metrics
- [ ] **Backtesting**: Strategy performance testing
- [ ] **Real-time Alerts**: Price and news monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only. It is not financial advice. Always do your own research and consult with financial professionals before making investment decisions. The authors are not responsible for any financial losses incurred from using this tool.**

## 🙏 Acknowledgments

- **OpenAI** for providing powerful language models
- **yfinance** for reliable stock data access
- **NewsAPI** for comprehensive news coverage
- **Python Community** for excellent data science libraries

## 📞 Support

- 📧 Email: yidong.chuang@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/sanqianpiao/ai-stock-analyzer/issues)
- 📖 Documentation: [Wiki](https://github.com/sanqianpiao/ai-stock-analyzer/wiki)

---

**Made with ❤️ and 🤖 by the AI Stock Analyzer Team**