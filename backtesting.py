"""Lightweight backtesting helpers for the EventQuant/Kalshi API.

This module wraps the public API endpoints, aligns candlestick series of
different lengths, and produces ML-ready feature matrices with convenient
train/val/test splits.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

DEFAULT_BASE_URL = "https://www.api.eventquant.com"


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _progress(seq: Iterable, desc: str, enabled: bool):
    if not enabled:
        for item in seq:
            yield item
        return

    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(seq, desc=desc)
    except Exception:
        for item in seq:
            yield item


def _avg(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


class EventQuantClient:
    """Simple HTTP client for the EventQuant API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 15,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch_markets(self, limit: int = 1000, offset: int = 0, **filters) -> Dict:
        """Fetch markets with filters identical to GET /markets."""
        params = {"limit": limit, "offset": offset, **filters}
        resp = self.session.get(
            f"{self.base_url}/markets", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_candlesticks(
        self,
        tickers: Sequence[str],
        *,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        period_interval: int = 1,
        get_event: bool = False,
        chunk_size: int = 40,
        progress: bool = True,
    ) -> Dict[str, List[Dict]]:
        """Fetch candlesticks for many tickers and collate by ticker."""
        payload_base: Dict[str, object] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
            "get_event": get_event,
        }

        out: Dict[str, List[Dict]] = {}
        for batch in _progress(
            list(_chunked(list(tickers), chunk_size)),
            desc="candles",
            enabled=progress,
        ):
            payload = {**payload_base, "tickers": batch}
            resp = self.session.post(
                f"{self.base_url}/markets/candlesticks",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            for entry in data.get("results", []):
                ticker = entry.get("ticker")
                if not ticker:
                    continue
                if entry.get("error"):
                    # Store an empty list to signal failure but keep keys consistent.
                    out.setdefault(ticker, [])
                    continue
                out[ticker] = entry.get("candlesticks") or []
        return out


def candles_to_price_tracks(
    candles: Sequence[Dict],
) -> Dict[str, List[Optional[float]]]:
    """Convert raw candle dicts to mid/high/low tracks."""
    mids: List[Optional[float]] = []
    highs: List[Optional[float]] = []
    lows: List[Optional[float]] = []

    for candle in candles:
        yes_ask = candle.get("yes_ask") or {}
        yes_bid = candle.get("yes_bid") or {}
        high = _avg([yes_ask.get("high"), yes_bid.get("high")])
        low = _avg([yes_ask.get("low"), yes_bid.get("low")])
        mid = _avg(
            [
                yes_ask.get("high"),
                yes_bid.get("high"),
                yes_bid.get("low"),
                yes_ask.get("low"),
            ]
        )
        mids.append(mid)
        highs.append(high)
        lows.append(low)

    return {"mid": mids, "high": highs, "low": lows}


def _most_common_length(lengths: Dict[str, int]) -> int:
    counter = Counter(lengths.values())
    target, _ = max(counter.items(), key=lambda pair: pair[1])
    return target


def _resize(
    seq: Sequence[Optional[float]],
    target_len: int,
    align: str,
    length_strategy: str,
    pad_value: Optional[float],
) -> List[Optional[float]]:
    if len(seq) == target_len:
        return list(seq)

    if len(seq) > target_len:
        return list(seq[-target_len:] if align == "right" else seq[:target_len])

    # len(seq) < target_len
    if length_strategy != "pad_to_longest":
        return list(seq)

    fill_value: Optional[float]
    if pad_value == "edge":
        fill_value = seq[-1] if align == "right" else seq[0]
    else:
        fill_value = pad_value

    pad_count = target_len - len(seq)
    padding = [fill_value] * pad_count
    return padding + list(seq) if align == "right" else list(seq) + padding


def normalize_lengths(
    features_by_ticker: Dict[str, Dict[str, List[Optional[float]]]],
    *,
    length_strategy: str = "trim_to_shortest",
    target_len: Optional[int] = None,
    align: str = "right",
    pad_value: Optional[float] = "edge",
) -> Tuple[
    Dict[str, Dict[str, List[Optional[float]]]], int, List[str]
]:
    """Align feature sequences to a common length.

    length_strategy:
      - "trim_to_shortest" (default): truncate longer series to the shortest length.
      - "pad_to_longest": pad shorter series to the longest length (or target_len).
      - "drop_incomplete": keep only series matching target_len (or the most common).
    """
    lengths = {
        ticker: len(next(iter(series.values()), []))
        for ticker, series in features_by_ticker.items()
    }
    if not lengths:
        return {}, 0, []

    if target_len is None:
        if length_strategy == "pad_to_longest":
            target_len = max(lengths.values())
        elif length_strategy == "drop_incomplete":
            target_len = _most_common_length(lengths)
        else:
            target_len = min(lengths.values())

    normalized: Dict[str, Dict[str, List[Optional[float]]]] = {}
    dropped: List[str] = []
    for ticker, series in features_by_ticker.items():
        series_len = lengths[ticker]
        if series_len == 0:
            dropped.append(ticker)
            continue

        if length_strategy == "drop_incomplete" and series_len != target_len:
            dropped.append(ticker)
            continue

        normalized[ticker] = {
            name: _resize(values, target_len, align, length_strategy, pad_value)
            for name, values in series.items()
        }

    return normalized, target_len, dropped


def build_feature_matrix(
    markets: Sequence[Dict],
    candles_by_ticker: Dict[str, List[Dict]],
    *,
    length_strategy: str = "trim_to_shortest",
    target_len: Optional[int] = None,
    align: str = "right",
    pad_value: Optional[float] = "edge",
    label_field: Optional[str] = "settlement_value",
    include_meta: bool = True,
) -> Dict[str, object]:
    """Create ML-ready feature matrix from markets + candles."""
    by_ticker = {m.get("ticker"): m for m in markets}
    features = {
        ticker: candles_to_price_tracks(candles)
        for ticker, candles in candles_by_ticker.items()
        if candles
    }

    aligned, final_len, dropped = normalize_lengths(
        features,
        length_strategy=length_strategy,
        target_len=target_len,
        align=align,
        pad_value=pad_value,
    )

    feature_names: List[str] = []
    if include_meta:
        feature_names.extend(["volume", "liquidity", "open_interest"])
    for i in range(final_len):
        feature_names.extend([f"mid_{i}", f"high_{i}", f"low_{i}"])

    X: List[List[Optional[float]]] = []
    y: List[Optional[float]] = []
    tickers_out: List[str] = []

    for ticker, series in aligned.items():
        row: List[Optional[float]] = []
        if include_meta:
            meta = by_ticker.get(ticker, {})
            row.extend(
                [
                    meta.get("volume"),
                    meta.get("liquidity"),
                    meta.get("open_interest"),
                ]
            )
        for i in range(final_len):
            row.extend(
                [
                    series["mid"][i],
                    series["high"][i],
                    series["low"][i],
                ]
            )
        X.append(row)
        tickers_out.append(ticker)
        if label_field:
            label_val = by_ticker.get(ticker, {}).get(label_field)
            y.append(label_val)

    return {
        "X": X,
        "y": y if label_field else None,
        "tickers": tickers_out,
        "feature_names": feature_names,
        "target_length": final_len,
        "dropped": dropped,
    }


def to_pandas(dataset: Dict[str, object]):
    """Convert a dataset from build_feature_matrix into a pandas DataFrame."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    df = pd.DataFrame(dataset["X"], columns=dataset["feature_names"])
    df.insert(0, "ticker", dataset["tickers"])
    if dataset.get("y") is not None:
        df["label"] = dataset["y"]
    return df


def train_val_test_split(
    dataset: Dict[str, object],
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    *,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Split a dataset dict into train/val/test partitions."""
    total = len(dataset.get("X", []))
    if total == 0:
        return ({"X": [], "y": [], "tickers": [], "feature_names": []},) * 3

    if round(train + val + test, 6) != 1.0:
        raise ValueError("train + val + test must equal 1.0")

    indices = list(range(total))
    if shuffle:
        random.seed(seed)
        random.shuffle(indices)

    train_end = int(total * train)
    val_end = train_end + int(total * val)

    def _slice(idx: List[int]) -> Dict[str, object]:
        return {
            "X": [dataset["X"][i] for i in idx],
            "y": [dataset["y"][i] for i in idx] if dataset.get("y") is not None else None,
            "tickers": [dataset["tickers"][i] for i in idx],
            "feature_names": dataset.get("feature_names", []),
            "target_length": dataset.get("target_length"),
        }

    return _slice(indices[:train_end]), _slice(indices[train_end:val_end]), _slice(
        indices[val_end:]
    )


# ---------------------------------------------------------------------------
# Quick strategy tests (momentum, mean reversion, Sharpe)
# ---------------------------------------------------------------------------

def _clean_prices(prices: Sequence[Optional[float]]) -> List[float]:
    return [p for p in prices if p is not None]


def backtest_momentum(
    mid_prices: Dict[str, Sequence[Optional[float]]],
    lookback_period: int = 5,
    entry_threshold: float = 0.01,
) -> Dict[str, float]:
    """Simple momentum: enter when pct change over lookback exceeds threshold."""
    ticker_returns: Dict[str, float] = {}
    for ticker, prices_raw in mid_prices.items():
        prices = _clean_prices(prices_raw)
        if len(prices) <= lookback_period:
            ticker_returns[ticker] = 0.0
            continue

        signal_idx = -1
        for i in range(lookback_period, len(prices)):
            base = prices[i - lookback_period]
            if base == 0:
                continue
            pct_change = (prices[i] - base) / base
            if pct_change > entry_threshold:
                signal_idx = i
                break

        if signal_idx == -1:
            ticker_returns[ticker] = 0.0
            continue

        entry_price = prices[signal_idx]
        exit_price = prices[-1]
        ticker_returns[ticker] = (exit_price - entry_price) / entry_price
    return ticker_returns


def backtest_buy_and_hold(
    mid_prices: Dict[str, Sequence[Optional[float]]],
    entry_index: int = 0,
) -> Dict[str, float]:
    """Baseline: buy at entry_index (default first) and exit at final candle."""
    ticker_returns: Dict[str, float] = {}
    for ticker, prices_raw in mid_prices.items():
        prices = _clean_prices(prices_raw)
        if len(prices) <= entry_index:
            ticker_returns[ticker] = 0.0
            continue
        entry_price = prices[entry_index]
        exit_price = prices[-1]
        if entry_price == 0:
            ticker_returns[ticker] = 0.0
            continue
        ticker_returns[ticker] = (exit_price - entry_price) / entry_price
    return ticker_returns


def backtest_breakout(
    mid_prices: Dict[str, Sequence[Optional[float]]],
    lookback_period: int = 20,
    breakout_threshold: float = 0.0,
) -> Dict[str, float]:
    """Enter on first breakout above prior lookback high by threshold; exit at end."""
    ticker_returns: Dict[str, float] = {}
    for ticker, prices_raw in mid_prices.items():
        prices = _clean_prices(prices_raw)
        if len(prices) <= lookback_period:
            ticker_returns[ticker] = 0.0
            continue

        signal_idx = -1
        window_max = max(prices[:lookback_period])
        for i in range(lookback_period, len(prices)):
            if window_max == 0:
                window_max = prices[i]
            if prices[i] > window_max * (1 + breakout_threshold):
                signal_idx = i
                break
            # update rolling max
            prev = prices[i - lookback_period]
            if prices[i] > window_max:
                window_max = prices[i]
            elif prev == window_max:
                window_max = max(prices[i - lookback_period + 1 : i + 1])

        if signal_idx == -1:
            ticker_returns[ticker] = 0.0
            continue

        entry_price = prices[signal_idx]
        exit_price = prices[-1]
        ticker_returns[ticker] = (exit_price - entry_price) / entry_price
    return ticker_returns


def backtest_moving_average_cross(
    mid_prices: Dict[str, Sequence[Optional[float]]],
    fast: int = 5,
    slow: int = 20,
) -> Dict[str, float]:
    """Enter on first fast MA cross above slow MA; exit at final candle."""
    if fast >= slow:
        raise ValueError("fast must be < slow for MA cross strategy")

    ticker_returns: Dict[str, float] = {}
    for ticker, prices_raw in mid_prices.items():
        prices = _clean_prices(prices_raw)
        if len(prices) <= slow:
            ticker_returns[ticker] = 0.0
            continue

        fast_sum = sum(prices[:fast])
        slow_sum = sum(prices[:slow])
        signal_idx = -1

        for i in range(slow, len(prices)):
            if i >= fast:
                fast_sum += prices[i] - prices[i - fast]
            slow_sum += prices[i] - prices[i - slow]

            fast_ma = fast_sum / fast
            slow_ma = slow_sum / slow
            prev_fast_ma = (fast_sum - prices[i] + prices[i - fast]) / fast
            prev_slow_ma = (slow_sum - prices[i] + prices[i - slow]) / slow

            if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
                signal_idx = i
                break

        if signal_idx == -1:
            ticker_returns[ticker] = 0.0
            continue

        entry_price = prices[signal_idx]
        exit_price = prices[-1]
        ticker_returns[ticker] = (exit_price - entry_price) / entry_price
    return ticker_returns


def backtest_mean_reversion(
    mid_prices: Dict[str, Sequence[Optional[float]]],
    ma_period: int = 10,
    entry_factor: float = 0.98,
) -> Dict[str, float]:
    """Mean reversion: enter when price falls below MA * entry_factor."""
    ticker_returns: Dict[str, float] = {}
    for ticker, prices_raw in mid_prices.items():
        prices = _clean_prices(prices_raw)
        if len(prices) <= ma_period:
            ticker_returns[ticker] = 0.0
            continue

        signal_idx = -1
        rolling_sum = sum(prices[:ma_period])
        for i in range(ma_period, len(prices)):
            ma = rolling_sum / ma_period
            if prices[i] < ma * entry_factor:
                signal_idx = i
                break
            # slide window
            rolling_sum += prices[i] - prices[i - ma_period]

        if signal_idx == -1:
            ticker_returns[ticker] = 0.0
            continue

        entry_price = prices[signal_idx]
        exit_price = prices[-1]
        ticker_returns[ticker] = (exit_price - entry_price) / entry_price
    return ticker_returns


def calculate_sharpe_ratio(
    returns: Sequence[float], risk_free_rate: float = 0.0
) -> float:
    """Sharpe ratio over non-zero returns."""
    traded = [r for r in returns if r != 0]
    if not traded:
        return 0.0

    mean_ret = sum(traded) / len(traded)
    if len(traded) == 1:
        std_dev = 0.0
    else:
        variance = sum((r - mean_ret) ** 2 for r in traded) / (len(traded) - 1)
        std_dev = variance**0.5

    if std_dev == 0:
        return float("inf") if mean_ret > risk_free_rate else 0.0
    return (mean_ret - risk_free_rate) / std_dev


def summarize_backtests(
    mid_prices: Dict[str, Sequence[Optional[float]]],
    *,
    risk_free_rate: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Run both canned strategies and return PnL + Sharpe summary."""
    strategies = {
        "buy_and_hold": backtest_buy_and_hold(mid_prices),
        "momentum": backtest_momentum(mid_prices),
        "mean_reversion": backtest_mean_reversion(mid_prices),
        "breakout": backtest_breakout(mid_prices),
        "ma_cross": backtest_moving_average_cross(mid_prices),
    }

    summary: Dict[str, Dict[str, float]] = {}
    for name, pnl in strategies.items():
        avg_ret = sum(pnl.values()) / len(pnl) if pnl else 0.0
        summary[name] = {
            "avg_return": avg_ret,
            "sharpe": calculate_sharpe_ratio(list(pnl.values()), risk_free_rate),
        }
    return summary
