"""Data access layer.

Uses akshare so the project can run on a free cloud runner without any paid
token. Every network call is retried because these public endpoints are not
always stable.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import requests

RETRY = 3
SLEEP_BASE = 2.0


@contextmanager
def network_timeout(seconds: float = 20):
    """Bound AKShare endpoints which do not expose a timeout parameter."""
    original = requests.sessions.Session.request
    def bounded(self, *args, **kwargs):
        kwargs['timeout'] = seconds
        return original(self, *args, **kwargs)
    requests.sessions.Session.request = bounded
    try:
        yield
    finally:
        requests.sessions.Session.request = original


def _retry(fn, label: str):
    last: Exception | None = None
    for attempt in range(RETRY):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - surface any provider error
            last = exc
            if attempt + 1 < RETRY:
                time.sleep(SLEEP_BASE * (attempt + 1))
    raise RuntimeError(f"{label} failed after {RETRY} attempts: {last}")


def fetch_constituents(index_code: str = "000300") -> pd.DataFrame:
    """Current index constituents with normalised weights.

    Returns columns: symbol (6-digit), name, weight, as_of.
    """
    raw = _retry(
        lambda: ak.index_stock_cons_weight_csindex(symbol=index_code),
        "index constituents",
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"empty constituent response for {index_code}")
    frame = pd.DataFrame(
        {
            "symbol": raw["成分券代码"].astype(str).str.zfill(6),
            "name": raw["成分券名称"].astype(str),
            "weight": pd.to_numeric(raw["权重"], errors="coerce"),
            "as_of": pd.to_datetime(raw["日期"]).max(),
        }
    )
    if (not np.isfinite(frame['weight']).all() or (frame['weight'] <= 0).any()
            or frame.symbol.duplicated().any() or frame.as_of.isna().any()):
        raise ValueError('Invalid constituent weights or dates; partial universe refused')
    if not frame.symbol.str.fullmatch(r'\d{6}').all():
        raise ValueError('Invalid stock code in constituents')
    frame["weight"] = frame["weight"] / frame["weight"].sum()
    return frame.reset_index(drop=True)


def fetch_spot() -> pd.DataFrame:
    """Whole-market snapshot; after the close this is the day's close.

    Returns columns: symbol, name, close, pct_chg, turnover.
    """
    raw = _retry(lambda: ak.stock_zh_a_spot_em(), "spot snapshot")
    return pd.DataFrame(
        {
            "symbol": raw["代码"].astype(str).str.zfill(6),
            "name": raw["名称"].astype(str),
            "close": pd.to_numeric(raw["最新价"], errors="coerce"),
            "pct_chg": pd.to_numeric(raw["涨跌幅"], errors="coerce") / 100.0,
            "turnover": pd.to_numeric(raw["换手率"], errors="coerce"),
            "circ_mv": pd.to_numeric(raw["流通市值"], errors="coerce"),
        }
    ).drop_duplicates("symbol")


def fetch_stock_history(
    symbols: list[str],
    days: int = 300,
    adjust: str = "hfq",
    pause: float = 0.15,
    start_date=None,
    end_date=None,
    workers: int = 1,
    timeout: float = 20,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Daily history for the given symbols.

    ``adjust='hfq'`` preserves the original research price/return convention.
    Its pct_chg differs from unadjusted returns; it is not share-account P&L.
    Use adjust='' when comparing dated quotes with the unadjusted spot feed.

    Returns columns: date, symbol, close, pct_chg, turnover.
    """
    frames: list[pd.DataFrame] = []
    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.now(tz='Asia/Shanghai').tz_localize(None).normalize()
    start = pd.Timestamp(start_date) if start_date is not None else end - pd.Timedelta(days=int(days * 1.7))
    failures: list[str] = []
    def fetch_one(symbol):
        try:
            raw = _retry(
                lambda s=symbol: ak.stock_zh_a_hist(
                    symbol=s,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust=adjust,
                    timeout=timeout,
                ),
                f"history {symbol}",
            )
        except Exception:  # noqa: BLE001
            return symbol, None
        if raw is None or raw.empty:
            return symbol, None
        result = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw["日期"]),
                    "symbol": symbol,
                    "close": pd.to_numeric(raw["收盘"], errors="coerce"),
                    "pct_chg": pd.to_numeric(raw["涨跌幅"], errors="coerce") / 100.0,
                    "turnover": pd.to_numeric(raw["换手率"], errors="coerce"),
                }
            )
        time.sleep(pause)
        return symbol, result
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        for symbol, frame in pool.map(fetch_one, sorted(set(symbols))):
            if frame is None:
                failures.append(symbol)
            else:
                frames.append(frame)
    if failures and not allow_partial:
        raise RuntimeError(f'历史行情下载不完整：{len(failures)} 只失败 {failures[:10]}，未提交状态')
    if not frames:
        out = pd.DataFrame(columns=["date", "symbol", "close", "pct_chg", "turnover"])
        out.attrs['failed_symbols'] = failures
        return out
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out.attrs['failed_symbols'] = failures
    return out


def fetch_index_history(index_code='000300', start_date=None, end_date=None):
    raw = _retry(lambda: ak.stock_zh_index_daily_em(
        symbol='sh' + index_code,
        start_date=pd.Timestamp(start_date).strftime('%Y%m%d'),
        end_date=pd.Timestamp(end_date).strftime('%Y%m%d')), 'official index history')
    frame = raw[['date', 'close']].copy()
    frame['date'] = pd.to_datetime(frame.date)
    frame['close'] = pd.to_numeric(frame.close, errors='coerce')
    if frame.empty or frame.date.duplicated().any() or not np.isfinite(frame.close).all() or (frame.close <= 0).any():
        raise ValueError('Invalid official index data')
    return frame.sort_values('date')


def load_industry(cache_path: Path, index_code: str = "000300") -> pd.Series:
    """Symbol -> SW level-1 industry.

    The akshare SW constituent endpoint is slow (one call per industry), so the
    mapping is cached in the repo and only refreshed when ``refresh=True`` is
    used by the maintenance script.
    """
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"industry cache missing at {cache_path}; run scripts/refresh_industry.py"
        )
    frame = pd.read_parquet(cache_path)
    frame['symbol'] = frame.symbol.astype(str).str.zfill(6)
    if frame.empty or frame.symbol.duplicated().any() or frame.industry.isna().any():
        raise ValueError('Invalid industry cache')
    return frame.set_index('symbol')['industry'].astype(str)


def refresh_industry(cache_path: Path, retry: int = 3) -> pd.DataFrame:
    """Fetch the SW level-1 mapping and write it to the cache.

    Kept in a separate function because it is a slow, low-frequency operation.
    """
    first = _retry(lambda: ak.sw_index_first_info(), "SW level-1 index list")
    records: list[dict[str, str]] = []
    for code, name in zip(first["行业代码"], first["行业名称"]):
        plain = str(code).split(".")[0]
        cons = None
        for attempt in range(retry):
            try:
                cons = ak.index_component_sw(symbol=plain)
                break
            except Exception:  # noqa: BLE001
                time.sleep(SLEEP_BASE * (attempt + 1))
        if cons is None:
            raise RuntimeError(f'SW industry {name} ({plain}) unavailable; existing cache retained')
        for symbol in cons["证券代码"].astype(str).str.zfill(6):
            records.append({"symbol": symbol, "industry": str(name)})
    frame = pd.DataFrame(records).drop_duplicates("symbol", keep="first")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if len(frame) < 300 or frame.industry.nunique() != len(first):
        raise ValueError('Incomplete industry refresh; existing cache retained')
    temporary = cache_path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(cache_path)
    return frame
