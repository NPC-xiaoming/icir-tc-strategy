"""Data access layer.

Uses akshare so the project can run on a free cloud runner without any paid
token. Every network call is retried because these public endpoints are not
always stable.
"""

from __future__ import annotations

import time
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
        requested=kwargs.get('timeout')
        kwargs['timeout'] = seconds if requested is None else (tuple(min(seconds,x) if x is not None else seconds for x in requested) if isinstance(requested,tuple) else min(seconds,requested))
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


def fetch_spot(history=None) -> pd.DataFrame:
    """Whole-market snapshot; after the close this is the day's close.

    Returns columns: symbol, name, close, pct_chg, turnover.
    """
    if history is not None:
        # Dated Sina raw close * outstanding shares gives float market cap in yuan.
        frame=history.sort_values('date').drop_duplicates('symbol',keep='last').copy()
        if not frame.empty and (not frame.provider.eq('sina').all() or not frame.adjust.eq('').all()):
            raise ValueError('Sina snapshot requires unadjusted Sina history')
        frame['name']=frame.symbol
        return frame[['symbol','name','close','pct_chg','turnover','circ_mv']]
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


def fetch_stock_history(*args, **kwargs):
    from .quotes import fetch_history
    return fetch_history(*args, **kwargs)


def fetch_index_history(index_code='000300', start_date=None, end_date=None, required_dates=None, provider=None):
    start=pd.Timestamp(start_date); end=pd.Timestamp(end_date)
    required=pd.DatetimeIndex(required_dates) if required_dates is not None else None
    providers=[
        ('eastmoney',lambda:ak.stock_zh_index_daily_em(symbol='sh'+index_code,
            start_date=start.strftime('%Y%m%d'),end_date=end.strftime('%Y%m%d'))),
        ('tencent',lambda:ak.stock_zh_index_daily_tx(symbol='sh'+index_code,
            start_date=start.strftime('%Y%m%d'),end_date=end.strftime('%Y%m%d'))),
        ('sina',lambda:ak.stock_zh_index_daily(symbol='sh'+index_code)),
    ]
    if provider is not None:
        if provider != 'sina': raise ValueError('Unsupported fixed index provider')
        providers=[item for item in providers if item[0]==provider]
    errors=[]
    for provider,fn in providers:
        def validated():
            raw=fn()
            frame=raw[['date','close']].copy()
            frame['date']=pd.to_datetime(frame.date)
            frame['close']=pd.to_numeric(frame.close,errors='coerce')
            frame=frame[frame.date.between(start,end)].sort_values('date')
            if (frame.empty or frame.date.duplicated().any() or not np.isfinite(frame.close).all()
                    or (frame.close<=0).any() or (required is not None and not required.isin(frame.date).all())):
                raise ValueError('official index dates/values incomplete')
            return frame
        try:
            frame=_retry(validated,f'index {provider}')
            frame.attrs['provider']=provider
            print(f'[index] source={provider} rows={len(frame)}',flush=True)
            return frame
        except Exception as exc:
            errors.append(f'{provider}: {exc}')
            print(f'[index] {errors[-1]}',flush=True)
    raise RuntimeError('All official index sources failed: '+'; '.join(errors))


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
