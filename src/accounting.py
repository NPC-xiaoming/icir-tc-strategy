"""Weight-based paper accounting, independent of signal and optimizer."""
from __future__ import annotations
import numpy as np
import pandas as pd


def dated_returns(history: pd.DataFrame, dates, symbols, allow_missing: bool = False) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    symbols = list(symbols)
    if not symbols:
        return pd.DataFrame(index=dates)
    if history.duplicated(['date', 'symbol']).any():
        raise ValueError('Duplicate dated stock quotes')
    wide = history.pivot(index='date', columns='symbol', values='pct_chg').reindex(index=dates, columns=symbols)
    values = wide.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1.).any():
        if allow_missing:
            return wide.fillna(0.0)
        missing = [(str(d.date()), s) for d, row in wide.iterrows() for s, v in row.items() if not np.isfinite(v)]
        raise ValueError(f'行情缺失或无效，不能当作停牌/零收益：{missing[:8]}')
    return wide


def drift(weights: pd.Series, returns: pd.Series) -> tuple[float, pd.Series]:
    if weights.empty:
        return 0., weights.copy()
    r = returns.reindex(weights.index)
    if not np.isfinite(r).all() or (r <= -1.).any():
        raise ValueError('Missing/invalid held-stock return')
    gross = float(weights @ r)
    return gross, weights * (1. + r) / (1. + gross)


def trading_cost(pretrade: pd.Series, target: pd.Series, costs: dict) -> tuple[float, float]:
    symbols = pretrade.index.union(target.index)
    delta = target.reindex(symbols, fill_value=0.) - pretrade.reindex(symbols, fill_value=0.)
    buy = float(delta.clip(lower=0).sum())
    sell = float(-delta.clip(upper=0).sum())
    buy_rate = (costs['commission_bps'] + costs['transfer_fee_bps'] + costs['slippage_bps']) / 10000.
    sell_rate = buy_rate + costs['stamp_duty_bps'] / 10000.
    return buy * buy_rate + sell * sell_rate, buy + sell
