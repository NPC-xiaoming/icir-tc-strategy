"""Factor computation and IC-IR signal synthesis (incremental, no look-ahead).

Factor definitions match the research version:
  momentum_120_20 = close[t-20] / close[t-120] - 1   (long winners)
  volatility_60   = 60-day realised vol, annualised  (long low vol)
  turnover_20     = 20-day average turnover rate     (long low turnover)

The composite is a weighted sum of direction-adjusted, industry-neutral
z-scores. Weights come from rolling IC-IR and are shifted by ``ic_lag`` so a
failing factor contributes nothing rather than a reversed view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL = 242
FACTOR_NAMES = ("momentum_120_20", "volatility_60", "turnover_20")
DIRECTIONS = {"momentum_120_20": 1.0, "volatility_60": -1.0, "turnover_20": -1.0}


def winsorize(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return series
    low, high = valid.quantile([lower, upper])
    return series.clip(lower=low, upper=high)


def zscore(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    std = series.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return series.where(series.isna(), 0.0)
    return (series - series.mean()) / std


def compute_raw_factors(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Raw factor values as of ``as_of``; index is symbol."""
    frame = history.loc[history["date"] <= pd.Timestamp(as_of)]
    records: list[dict[str, object]] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        ordered = group.sort_values("date")
        close = ordered["close"].to_numpy(dtype=float)
        ret = ordered["pct_chg"].to_numpy(dtype=float)
        turnover = ordered["turnover"].to_numpy(dtype=float)
        if len(close) < 121:
            continue
        base, recent = close[-121], close[-21]
        if not np.isfinite(base) or not np.isfinite(recent) or base <= 0:
            continue
        records.append(
            {
                "symbol": symbol,
                "momentum_120_20": recent / base - 1.0,
                "volatility_60": (
                    float(np.nanstd(ret[-60:], ddof=1) * np.sqrt(ANNUAL)) if len(ret) >= 60 else np.nan
                ),
                "turnover_20": float(np.nanmean(turnover[-20:])) if len(turnover) >= 20 else np.nan,
            }
        )
    if not records:
        return pd.DataFrame(columns=list(FACTOR_NAMES)).rename_axis("symbol")
    return pd.DataFrame(records).set_index("symbol")


def neutral_zscore(
    raw: pd.DataFrame,
    industry: pd.Series,
    min_industry_size: int = 5,
) -> pd.DataFrame:
    """Industry-neutral winsorised z-scores; small industries fall back to market."""
    out = pd.DataFrame(index=raw.index, columns=list(FACTOR_NAMES), dtype=float)
    labels = industry.reindex(raw.index).fillna("其他").astype(str)
    for factor in FACTOR_NAMES:
        values = pd.to_numeric(raw[factor], errors="coerce")
        market_z = zscore(winsorize(values))
        group_z = pd.Series(np.nan, index=raw.index, dtype=float)
        for _, idx in labels.groupby(labels).groups.items():
            if len(idx) < min_industry_size:
                continue
            group_z.loc[idx] = zscore(winsorize(values.loc[idx]))
        out[factor] = group_z.where(group_z.notna(), market_z)
    return out


def composite(z: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Direction-adjusted weighted composite score."""
    score = pd.Series(0.0, index=z.index, dtype=float)
    valid = pd.Series(True, index=z.index)
    total = 0.0
    for factor in FACTOR_NAMES:
        weight = float(weights.get(factor, 0.0))
        if weight <= 0:
            continue
        total += weight
        series = pd.to_numeric(z[factor], errors="coerce")
        valid &= series.notna()
        score = score + weight * DIRECTIONS[factor] * series.fillna(0.0)
    if total <= 0:
        return pd.Series(np.nan, index=z.index, dtype=float)
    return (score / total).where(valid)


def compute_ic(z: pd.DataFrame, forward_returns: pd.Series) -> dict[str, float]:
    """Cross-sectional IC of each factor against realised forward returns."""
    out: dict[str, float] = {}
    fwd = pd.to_numeric(forward_returns.reindex(z.index), errors="coerce")
    for factor in FACTOR_NAMES:
        values = pd.to_numeric(z[factor], errors="coerce")
        mask = values.notna() & fwd.notna()
        if mask.sum() < 30:
            out[factor] = float("nan")
            continue
        corr = np.corrcoef(values[mask], fwd[mask])[0, 1]
        out[factor] = float(corr) if np.isfinite(corr) else float("nan")
    return out


def icir_weights(
    ic_history: pd.DataFrame,
    window: int = 24,
    lag: int = 1,
) -> dict[str, float]:
    """Rolling IC-IR weights over already-realised ICs.

    ``ic_history`` must contain only ICs that were measurable before the
    current rebalance. ``lag`` skips that many additional most-recent
    observations, reproducing the research version's extra execution buffer.
    ICs are floored at zero after averaging, so a factor whose recent
    information coefficient is negative receives no weight instead of a
    reversed one.
    """
    if ic_history is None or ic_history.empty:
        return {factor: 1.0 / len(FACTOR_NAMES) for factor in FACTOR_NAMES}
    frame = ic_history.reindex(columns=list(FACTOR_NAMES)).astype(float)
    usable = frame.iloc[: max(0, len(frame) - lag)] if lag > 0 else frame
    recent = usable.tail(window)
    weights: dict[str, float] = {}
    min_periods = max(3, window // 2)
    for factor in FACTOR_NAMES:
        series = recent[factor].dropna()
        if len(series) < min_periods:
            weights[factor] = 0.0
            continue
        std = series.std(ddof=1)
        weights[factor] = float(series.mean() / std) if std and std > 0 else 0.0
    positive = {k: max(0.0, v) for k, v in weights.items()}
    total = sum(positive.values())
    if total <= 0:
        return {factor: 1.0 / len(FACTOR_NAMES) for factor in FACTOR_NAMES}
    return {k: v / total for k, v in positive.items()}


def forward_return(
    history: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Holding-period return per symbol between two rebalance dates."""
    window = history.loc[(history["date"] > pd.Timestamp(start)) & (history["date"] <= pd.Timestamp(end))]
    if window.empty:
        return pd.Series(dtype=float)
    return window.groupby("symbol")["pct_chg"].apply(lambda s: float((1.0 + s).prod() - 1.0))
