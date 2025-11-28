# EventQuant Backtesting SDK

A minimal Python helper to pull markets and candlesticks from
`https://www.api.eventquant.com`, align candle series of different lengths, and
produce ML-ready feature matrices with train/val/test splits.

The SDK is intentionally lightweight and only depends on `requests`. If
installed, `pandas` is used to build a convenience DataFrame and `tqdm` is used
for progress bars. Suggested extras for ML examples below:

```bash
pip install pandas tqdm scikit-learn torch
```

## Quick start

```python
from sdk.backtesting import (
    EventQuantClient,
    build_feature_matrix,
    to_pandas,
    train_val_test_split,
)

# 1) Pull markets
client = EventQuantClient()  # base_url defaults to https://www.api.eventquant.com
markets = client.fetch_markets(
    limit=1000,
    ticker_regex="KXBTC-",
    title_regex="range",
    min_duration_minutes=50,
    max_duration_minutes=60,
    volume__gt=1000,
    order_by="close_time",
)

tickers = [m["ticker"] for m in markets["items"]]

# 2) Fetch candlesticks (period_interval in minutes)
candles = client.fetch_candlesticks(
    tickers,
    period_interval=1,
    get_event=False,
    progress=True,
)

# 3) Build aligned feature matrix
dataset = build_feature_matrix(
    markets["items"],
    candles,
    length_strategy="drop_incomplete",  # keep only tickers that match target_len
    target_len=60,  # enforce exact candle count like your previous "exact=60"
    align="right",  # align on the most recent candles
    pad_value="edge",  # ignored for drop_incomplete, used when padding
    label_field="settlement_value",
)

# 4) Optional: DataFrame + splits for ML
df = to_pandas(dataset)  # requires pandas; returns None if pandas is missing
train, val, test = train_val_test_split(dataset, train=0.7, val=0.15, test=0.15)
```

## End-to-end workflow (markets → features → splits)

```python
from sdk.backtesting import (
    EventQuantClient,
    build_feature_matrix,
    to_pandas,
    train_val_test_split,
)

client = EventQuantClient()
markets = client.fetch_markets(limit=500, volume__gt=1000, order_by="close_time")
tickers = [m["ticker"] for m in markets["items"]]
candles = client.fetch_candlesticks(tickers, period_interval=1, progress=True)

dataset = build_feature_matrix(
    markets["items"],
    candles,
    length_strategy="pad_to_longest",
    align="right",
    pad_value="edge",
    label_field="settlement_value",
)
train, val, test = train_val_test_split(dataset)
df = to_pandas(dataset)  # optional DataFrame view
```

## Quick PnL/Sharpe sanity checks

Use the prebuilt toy strategies to smoke-test your candle features:

```python
from sdk.backtesting import (
    candles_to_price_tracks,
    backtest_momentum,
    backtest_mean_reversion,
    backtest_buy_and_hold,
    backtest_breakout,
    backtest_moving_average_cross,
    summarize_backtests,
)

# mid prices from the raw candlestick payload
mid_prices = {
    ticker: candles_to_price_tracks(candles)["mid"]
    for ticker, candles in candles.items()
}

momentum_pnl = backtest_momentum(mid_prices, lookback_period=5, entry_threshold=0.01)
mean_rev_pnl = backtest_mean_reversion(mid_prices, ma_period=10, entry_factor=0.98)
buy_hold_pnl = backtest_buy_and_hold(mid_prices)
breakout_pnl = backtest_breakout(mid_prices, lookback_period=20, breakout_threshold=0.0)
ma_cross_pnl = backtest_moving_average_cross(mid_prices, fast=5, slow=20)
summary = summarize_backtests(mid_prices)  # returns avg_return and Sharpe per strategy
```

Included strategies:
- `buy_and_hold`: baseline from entry index (default first candle) to last.
- `momentum`: enter when % change over `lookback_period` exceeds `entry_threshold`.
- `mean_reversion`: enter when price drops below MA * `entry_factor`.
- `breakout`: enter on first close above the prior `lookback_period` high.
- `ma_cross`: enter on first fast MA cross above slow MA.

All helpers consume `Dict[str, List[float]]` of mid-prices and ignore `None`
values. Sharpe ignores zero trades and returns `inf` if returns are constant and
positive.

## Modeling examples

### Logistic classifier (scikit-learn)

Treat settlement value (or any numeric field) as a binary label and train a
simple classifier on flattened candle features:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sdk.backtesting import train_val_test_split, build_feature_matrix, EventQuantClient

client = EventQuantClient()
markets = client.fetch_markets(limit=1000, volume__gt=1000)
tickers = [m["ticker"] for m in markets["items"]]
candles = client.fetch_candlesticks(tickers, period_interval=1, progress=True)

dataset = build_feature_matrix(
    markets["items"],
    candles,
    length_strategy="drop_incomplete",
    target_len=60,
    label_field="settlement_value",
)
train, val, test = train_val_test_split(dataset)

X_train = np.array(train["X"])
y_train = (np.array(train["y"]) > 0.5).astype(int)  # example threshold
X_val = np.array(val["X"])
y_val = (np.array(val["y"]) > 0.5).astype(int)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
val_pred = clf.predict_proba(X_val)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, val_pred))
```

### Sequence model (PyTorch RNN/LSTM)

Use the aligned sequences directly by reshaping back to `[batch, time, features]`.
Here we only keep the `mid` price per step for a tiny demo RNN:

```python
import torch
import torch.nn as nn
from sdk.backtesting import build_feature_matrix, train_val_test_split, EventQuantClient

client = EventQuantClient()
markets = client.fetch_markets(limit=500)
tickers = [m["ticker"] for m in markets["items"]]
candles = client.fetch_candlesticks(tickers, period_interval=1, progress=True)

dataset = build_feature_matrix(
    markets["items"],
    candles,
    length_strategy="drop_incomplete",
    target_len=60,
    include_meta=False,  # keep only time-series values
    label_field="settlement_value",
)
train, val, _ = train_val_test_split(dataset)

def to_tensor(split):
    # Extract only mid prices: every third feature in the flattened vector
    seq_len = dataset["target_length"]
    mid_only = [
        [row[i] for i in range(0, 3 * seq_len, 3)]
        for row in split["X"]
    ]
    X = torch.tensor(mid_only, dtype=torch.float32).unsqueeze(-1)  # [B, T, 1]
    y = torch.tensor(split["y"], dtype=torch.float32).unsqueeze(-1)
    return X, y

X_train, y_train = to_tensor(train)
X_val, y_val = to_tensor(val)

model = nn.Sequential(
    nn.LSTM(input_size=1, hidden_size=16, batch_first=True),
    nn.Flatten(),
    nn.Linear(16 * dataset["target_length"], 1),
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    pred, _ = model[0](X_train)
    out = model[1:](pred)
    loss = criterion(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        val_pred, _ = model[0](X_val)
        val_out = model[1:](val_pred)
        val_loss = criterion(val_out, y_val)
    print(f"epoch {epoch} train_loss={loss.item():.4f} val_loss={val_loss.item():.4f}")
```

Replace the head with your preferred architecture (GRU/Transformer) or add
conditioning features (volume/liquidity) by keeping `include_meta=True` and
reshaping accordingly.

## Candle length handling

`build_feature_matrix` accepts three strategies:

- `trim_to_shortest` (default): truncate longer series to the shortest length.
- `pad_to_longest`: pad shorter series to the longest (or `target_len`) using
  `pad_value` (use `"edge"` to repeat the edge candle).
- `drop_incomplete`: keep only series with `target_len` candles (or the most
  common length if `target_len` is omitted).

Set `align="right"` to keep the most recent candles or `"left"` to keep the
earliest.

## Feature calculation

For each candle the SDK computes:

- `mid`: mean of `yes_ask.high`, `yes_bid.high`, `yes_bid.low`, `yes_ask.low`.
- `high`: mean of `yes_ask.high` and `yes_bid.high`.
- `low`: mean of `yes_ask.low` and `yes_bid.low`.

The resulting feature matrix flattens `[mid, high, low]` for every time step.
When `include_meta=True` (default) the first three columns are
`volume, liquidity, open_interest`. Labels default to `settlement_value` but can
be switched via `label_field`.
