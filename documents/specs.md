# AI Stock Analyzer Project Specification

## Project Overview
### Title
AI Stock Analyzer: A Generative AI-Powered Tool for Stock Data Retrieval, Analysis, and Insights

### Description
This project builds a Python-based application that leverages Generative AI (using OpenAI's GPT models as the LLM backbone) to analyze stocks. It combines data fetching, basic quantitative analysis, sentiment analysis from news (via RAG principles), and an AI agent for generating actionable recommendations. The tool will be modular, extensible, and deployable as a CLI or simple web app.

Key Gen AI Applications Demonstrated:
- **AI Agents**: An autonomous agent that orchestrates tasks like data fetching, analysis, and recommendation generation.
- **RAG (Retrieval-Augmented Generation)**: Retrieves real-time news/articles and augments GPT prompts for grounded, up-to-date sentiment analysis to reduce hallucinations.
- **Prompt Engineering**: Structured prompts for GPT to ensure consistent, reliable outputs.

### Goals
- **Learning Objectives**: Systematically apply Gen AI concepts (agents, RAG) in a real-world finance scenario.
- **Functional Goals**: Provide users with stock insights (e.g., trend, sentiment, buy/sell/hold recommendation) via a simple interface.
- **Non-Functional Goals**: Modular code for easy extension; handle errors gracefully; respect API rate limits; ensure data privacy (no storage of sensitive info).

### Target Users
- Beginner/intermediate developers learning Gen AI.
- Hobby investors seeking quick AI-assisted analysis.
- extensible for advanced users (e.g., add portfolio tracking).

### Assumptions & Constraints
- **LLM**: Use OpenAI GPT-4o-mini (cost-effective, fast) or GPT-4o for complex reasoning. User has OpenAI API key.
- **Environment**: Python 3.10+; no internet installs in runtime (all deps pre-installed).
- **Data Sources**: Free APIs (yfinance for stocks, NewsAPI for news). No paid services unless specified.
- **Scope Limits**: Analyze single stocks initially; no real-time trading execution; mock data for testing.
- **Ethical Considerations**: Include disclaimers (e.g., "Not financial advice"); cite sources in outputs.

## Requirements
### Functional Requirements
1. **Data Fetching Module**:
   - Fetch historical stock data (e.g., OHLCV) for a given ticker and period (default: 1 month).
   - Fetch recent news headlines (top 5-10) related to the ticker.
   - Output: Pandas DataFrames or JSON for easy integration.

2. **Quantitative Analysis Module**:
   - Compute basic metrics: Closing price, 5/20-day moving averages, RSI (Relative Strength Index), volume trends.
   - Detect simple trends (e.g., bullish/bearish based on MA crossover).
   - Output: Summary dict with metrics and trend label.

3. **Sentiment Analysis Module (RAG-Enabled)**:
   - Retrieve news via API.
   - Embed headlines (using OpenAI embeddings) and store in a simple vector store (e.g., FAISS).
   - Use GPT to generate sentiment summary (positive/negative/neutral) augmented with retrieved context.
   - Output: Sentiment score (-1 to 1) and explanation.

4. **AI Agent Orchestrator**:
   - Use LangChain to build a ReAct-style agent with tools for the above modules.
   - Agent prompt: "Analyze [ticker]: Fetch data, compute metrics, retrieve news, analyze sentiment, then recommend buy/sell/hold with reasoning."
   - Handle user queries conversationally (e.g., follow-up questions).
   - Output: Structured response (e.g., JSON with sections: data_summary, analysis, recommendation).

5. **User Interface**:
   - CLI: Command-line input for ticker/period.
   - Optional: Streamlit web app for interactive dashboard (visualize charts with Plotly).
   - Logging: Verbose mode for debugging agent steps.

6. **Reporting & Visualization**:
   - Generate a markdown/HTML report with tables/charts.
   - Include citations for news sources.

### Non-Functional Requirements
- **Performance**: <10s per analysis; cache data for 5min.
- **Security**: Store API keys in env vars (.env file); no hardcoding.
- **Testing**: Unit tests for each module (80% coverage); integration tests for agent.
- **Documentation**: Inline docstrings; README.md with setup/run instructions.
- **Extensibility**: Abstract LLM calls for easy swap (e.g., to Grok later).
- **Error Handling**: Graceful fallbacks (e.g., mock data if API fails).

### Input/Output Examples
- **Input**: `python analyzer.py --ticker AAPL --period 1mo --verbose`
- **Output** (Console/Markdown):
  ```
  ## AAPL Analysis (as of 2025-09-20)

  ### Data Summary
  | Metric | Value |
  |--------|-------|
  | Latest Close | $220.50 |
  | 5-Day MA | $218.20 |
  | RSI | 65.3 |

  ### Trend: Bullish (Close > MA)

  ### Sentiment: Positive (Score: 0.7)
  Reasons: Recent product launches; extracted from [NewsAPI source 1], [source 2].

  ### Recommendation: Buy
  Reasoning: Strong upward trend + positive news; potential 5-10% upside in next month.
  ```

## Project Structure
Organize as a Python package for modularity. Use GitHub Copilot to generate code snippets based on this spec (e.g., prompt: "Implement fetch_stock_data function per spec").

```
ai_stock_analyzer/
├── README.md                 # Setup, usage, examples
├── requirements.txt          # Deps: yfinance, pandas, openai, langchain, langchain-openai, faiss-cpu, newsapi-python, streamlit, plotly, pytest
├── .env.example              # Template for API keys
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py             # Load env vars, constants (e.g., NEWSAPI_KEY)
│   ├── data_fetcher.py       # fetch_stock_data, fetch_news
│   ├── analyzer.py           # quantitative_analysis, sentiment_analysis (with RAG)
│   ├── agent.py              # LangChain agent setup and run
│   └── reporter.py           # Generate reports/visuals
├── tests/
│   ├── test_data_fetcher.py
│   ├── test_analyzer.py
│   └── test_agent.py
├── app.py                    # Streamlit entrypoint (optional)
├── main.py                   # CLI entrypoint
└── mocks/                    # Mock data for testing (e.g., sample_stock_data.json)
```

## Development Steps
Follow this phased approach. Use GitHub Copilot by copying spec sections into comments/prompts (e.g., "# Per spec: Implement RAG in sentiment_analysis").

### Phase 1: Setup & Data Fetching (1-2 days)
1. Initialize repo: `git init`, add structure above.
2. Install deps: `pip install -r requirements.txt`.
3. Implement `data_fetcher.py`: Test with AAPL; add mocks.
4. Write unit tests; run `pytest`.
5. Copilot Prompt Example: "Write a function to fetch stock history using yfinance, returning a Pandas DF, with error handling for invalid tickers."

### Phase 2: Core Analysis (2 days)
1. Implement `analyzer.py`: Quantitative metrics first, then basic sentiment (non-RAG).
2. Integrate OpenAI: Test simple GPT call for sentiment.
3. Add RAG: Use OpenAI embeddings + FAISS for retrieval.
4. Tests: Mock API responses.
5. Copilot Prompt: "Add RSI calculation to analyze_stock; use TA-Lib if available, else pandas."

### Phase 3: AI Agent & Orchestration (2-3 days)
1. Implement `agent.py`: Define tools, initialize LangChain agent with GPT.
2. Test end-to-end: Run agent on sample query.
3. Handle chaining: Agent calls modules sequentially.
4. Copilot Prompt: "Build a LangChain ReAct agent with tools for stock fetch and sentiment; use GPT-4o-mini."

### Phase 4: UI, Reporting & Polish (1-2 days)
1. Implement `reporter.py`: Markdown output + Plotly charts.
2. Build CLI in `main.py` (use argparse).
3. Optional: Streamlit app.
4. Full integration tests; add logging (logging module).
5. Documentation: Update README with badges (e.g., pytest coverage).
6. Copilot Prompt: "Generate Streamlit dashboard for stock analysis with Plotly candlestick chart."

### Phase 5: Iteration & Deployment (Ongoing)
1. Deploy: To GitHub Pages (for static reports) or Heroku (for app).
2. Extensions: Add multi-stock comparison, voice input (if using Grok voice later).
3. Review: Run full analysis; refine prompts based on outputs.
4. Share: Push to GitHub; invite feedback.

## Testing Strategy
- **Unit**: Pytest for isolated functions (e.g., `test_analyze_stock` with mocked data).
- **Integration**: Mock OpenAI/NewsAPI with `unittest.mock`.
- **E2E**: Run CLI with real APIs; assert output structure.
- **Edge Cases**: Invalid ticker, API downtime, low-volume stocks.

## Risks & Mitigations
- **API Costs**: Use GPT-4o-mini; limit news fetches.
- **Hallucinations**: Enforce RAG; validate outputs with rules (e.g., score thresholds).
- **Data Accuracy**: Disclaimer + source citations.
- **Copilot Usage**: Review generated code for spec adherence; avoid over-reliance.

This spec serves as your "source of truth." Paste sections into GitHub Copilot chats for targeted code gen (e.g., in VS Code: Ctrl+I). What's next—start Phase 1, tweak a section, or generate initial code for a module? Let's iterate!