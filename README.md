# Quantile-Based Regime Markov Model for Financial Returns

This project implements a **quantile-based regime Markov model** for analyzing financial time series (stocks and cryptocurrencies).

Unlike latent regime models, this approach explicitly defines market regimes by discretizing log-returns into equiprobable quantile bins. A first-order Markov chain is then fitted to the resulting state sequence to capture transition dynamics. The model is particularly useful for:

- Market regime identification and analysis
- Statistical testing of serial dependence in returns
- Feature engineering for machine learning and deep learning models
- Enhancing interpretability and robustness in quantitative research

## Core Methodology

1. Download OHLCV price data (stocks or cryptocurrencies)
2. Compute log-returns
3. Discretize returns into quantile-based regimes (equiprobable bins)
4. Estimate the Markov transition matrix
5. Perform a chi-squared test of independence on consecutive states
6. Select the smallest number of states yielding statistically significant dependence (p-value < 0.05)
7. Visualize the transition structure

## Key Advantages of Quantile-Based Regimes

- Equiprobable states ensure sufficient observations in each regime
- Non-parametric: no assumptions of Gaussianity or stationarity
- Robust to outliers and heavy-tailed return distributions
- Scale-free and comparable across different assets

Each regime represents a relative market condition (e.g., strongly negative, neutral, strongly positive).

## Markov Chain Assumptions

The model assumes a first-order Markov property on the discretized states:

$$P(S_{t+1} \mid S_t, S_{t-1}, \dots) = P(S_{t+1} \mid S_t)$$

This enables analysis of:

- Regime persistence
- Asymmetric transition behavior (bull vs. bear markets)
- Return clustering effects

**Important**: States are **observable** and deterministically assigned based on returns. This is **not** a Hidden Markov Model (HMM).

| Aspect                  | This Model                  | Hidden Markov Model (HMM) |
|-------------------------|-----------------------------|---------------------------|
| States                  | Observable (quantile-based) | Hidden / latent           |
| Emission model          | None                        | Required                  |
| State inference         | Deterministic               | Probabilistic (Baum-Welch)|
| Parameter estimation    | Direct counting             | EM algorithm              |

## Statistical Validation

A chi-squared test is used to assess whether transitions are independent:

- **H₀**: $$\(S_{t+1} \perp S_t\)$$ (no serial dependence in regimes)  
- **H₁**: $$\(S_{t+1} \not\perp S_t\)$$ (serial dependence in regimes)

A significant result indicates structural dependence and deviation from pure randomness expected under strict forms of the Efficient Market Hypothesis.

## State Selection

The algorithm incrementally increases the number of states starting from 2, stopping at the smallest number where the chi-squared test rejects independence (p < 0.05). This balances detection power with avoidance of over-discretization and data fragmentation.

## Applications in Machine Learning

Regime outputs serve as powerful features:

- Current regime (categorical or one-hot encoded)
- Transition probabilities from current state
- Regime persistence probability
- Stationary distribution metrics
- Transition matrix entropy
- Regime switching frequency

These features often improve performance in forecasting, volatility modeling, and regime-conditioned strategies due to noise reduction and structural signal extraction.

## Project Structure
├── main.py              # Command-line interface and workflow orchestration\
├── markov.py            # Regime discretization, transition matrix estimation, and chi-squared testing\
├── data_downloader.py   # OHLCV data retrieval (yfinance for stocks, python-binance for crypto)\
├── visualizer.py        # Markov chain visualization using Graphviz and NetworkX\
├── mc.dot               # Generated Graphviz DOT file (output)\
└── mc_plot.png          # Rendered Markov chain graph (output)\

Matrices are printed to console.

## Installation

Install the required Python dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn yfinance python-binance networkx pydot graphviz
```

## Usage

Run the model from the command line:
```bash
`python main.py \
  --market STOCK \          # Choices: STOCK or CRYPTO
  --asset TSLA \            # Ticker/symbol, e.g., TSLA, AAPL, BTCUSDT
  --start 2023-01-01 \      # Start date (YYYY-MM-DD)
  --end 2024-01-01 \        # End date (YYYY-MM-DD)
  --timefreq 1d \           # Time frequency (e.g., 1d, 1h, 5m – subject to data source limits)
  --max_state 5            # Optional: maximum number of regimes to test (default: 5)`
```
