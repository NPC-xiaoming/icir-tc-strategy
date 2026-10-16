"""Dated stock quotes with persistent cache and checked provider fallback.

Never splice adjusted prices from different providers to calculate a return.
Eastmoney's rounded HFQ return is the existing strategy convention; a backup
must match at least three overlapping Eastmoney returns within one basis point.
"""
from __future__ import annotations
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import akshare as ak
import numpy as np
import pandas as pd

COLUMNS = ['date','symbol','close','pct_chg','turnover','provider','adjust']
PROVIDERS = ('eastmoney','tencent','sina')
RETURN_TOLERANCE = .0001


def _provider_frame(provider, symbol, start, end, adjust, timeout, calendar=None):
    market = ('sh' if symbol.startswith('6') else 'sz') + symbol
    # Include earlier closes; deriving the first requested day's return needs an anchor.
    begin = (start-pd.Timedelta(days=40)).strftime('%Y%m%d')
    finish = end.strftime('%Y%m%d')
    if provider == 'eastmoney':
        raw=ak.stock_zh_a_hist(symbol=symbol,period='daily',start_date=begin,
                               end_date=finish,adjust=adjust,timeout=timeout)
        if raw is None or raw.empty: raise ValueError('empty response')
        out=raw.rename(columns={'日期':'date','收盘':'close','涨跌幅':'pct_chg','换手率':'turnover'})
        out=out[['date','close','pct_chg','turnover']].copy()
        out['pct_chg']=pd.to_numeric(out.pct_chg,errors='coerce')/100.
    else:
        if provider=='tencent':
            raw=ak.stock_zh_a_hist_tx(symbol=market,start_date=begin,end_date=finish,adjust=adjust,timeout=timeout)
        else:
            raw=ak.stock_zh_a_daily(symbol=market,start_date=begin,end_date=finish,adjust=adjust)
        if raw is None or raw.empty: raise ValueError('empty response')
        out=raw.copy(); out['date']=pd.to_datetime(out.date)
        out=out.sort_values('date')
        out['close']=pd.to_numeric(out.close,errors='coerce')
        out['pct_chg']=out.close.pct_change(fill_method=None)
        # A return across multiple trading sessions cannot stand for one day.
        if calendar is not None:
            positions=pd.Series(np.arange(len(calendar)),index=pd.DatetimeIndex(calendar))
            steps=out.date.map(positions).diff()
            out.loc[steps!=1,'pct_chg']=np.nan
        out['turnover']=pd.to_numeric(out['turnover'],errors='coerce')*100. if 'turnover' in out else np.nan
    out['date']=pd.to_datetime(out.date,errors='coerce')
    out['symbol']=symbol
    out['close']=pd.to_numeric(out.close,errors='coerce')
    out['turnover']=pd.to_numeric(out.turnover,errors='coerce')
    out=out[out.date.notna() & (out.date<=end)].copy()
    if out.date.duplicated().any(): raise ValueError('duplicate quote dates')
    valid=np.isfinite(out.close)&(out.close>0)&np.isfinite(out.pct_chg)&(out.pct_chg>-1)
    return out.loc[valid,['date','symbol','close','pct_chg','turnover']].sort_values('date')


def fetch_history(symbols, days=300, adjust='hfq', pause=.15, start_date=None,
                  end_date=None, workers=2, timeout=12, allow_partial=False,
                  cache_path=None, required_dates=None, calendar=None, require_turnover=False):
    end=pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.now(tz='Asia/Shanghai').tz_localize(None).normalize()
    start=pd.Timestamp(start_date) if start_date is not None else end-pd.Timedelta(days=int(days*1.7))
    required=pd.DatetimeIndex(required_dates) if required_dates is not None else pd.bdate_range(start,end)
    cached=pd.read_parquet(cache_path) if cache_path and Path(cache_path).is_file() else pd.DataFrame(columns=COLUMNS)
    if not cached.empty: cached['date']=pd.to_datetime(cached.date)
    lock=threading.Lock(); unavailable=set(); provider_failures={p:0 for p in PROVIDERS}
    errors=[]; writes=[]; sources={}; failed=[]; results=[]

    def eligible(frame):
        if frame.empty: return frame
        valid=np.isfinite(frame.close)&(frame.close>0)&np.isfinite(frame.pct_chg)&(frame.pct_chg>-1)
        if require_turnover: valid &= np.isfinite(frame.turnover)&(frame.turnover>=0)
        return frame.loc[valid & frame.date.isin(required)].copy()

    def one(symbol):
        local=cached[(cached.symbol==symbol)&(cached.adjust==adjust)]
        reference=local[local.provider=='eastmoney'].copy()
        best=pd.DataFrame(columns=COLUMNS)
        best_source=None
        for provider in PROVIDERS:
            old=local[local.provider==provider].copy()
            subset=eligible(old)
            if len(subset)>len(best): best,best_source=subset,provider
            if len(subset)==len(required): return symbol,subset,provider,[] ,[]
        events=[]; updates=[]
        for provider in PROVIDERS:
            old=local[local.provider==provider].copy()
            with lock: disabled=provider in unavailable
            if disabled:
                events.append({'symbol':symbol,'provider':provider,'error':'provider circuit open after repeated request failures'})
                continue
            fresh=None
            for attempt in range(2):
                try:
                    time.sleep(pause+random.uniform(0,.1))
                    fresh=_provider_frame(provider,symbol,start,end,adjust,timeout,calendar=calendar)
                    if fresh.empty: raise ValueError('no valid dated returns')
                    break
                except Exception as exc:
                    reason=f'{type(exc).__name__}: {str(exc)[:400]}'
                    events.append({'symbol':symbol,'provider':provider,'attempt':attempt+1,'error':reason})
                    print(f'[quote] {symbol} {provider} attempt={attempt+1} {reason}',flush=True)
                    if attempt==0: time.sleep(.5+random.uniform(0,.25))
            if fresh is None or fresh.empty:
                with lock:
                    provider_failures[provider]+=1
                    if provider_failures[provider]>=3: unavailable.add(provider)
                continue
            # Responses are validated before accepting them into the durable cache.
            fresh=fresh.copy(); fresh['provider']=provider; fresh['adjust']=adjust
            if provider=='eastmoney':
                reference=pd.concat([reference,fresh],ignore_index=True).drop_duplicates('date',keep='last')
            else:
                overlap=reference[['date','pct_chg']].merge(fresh[['date','pct_chg']],on='date',suffixes=('_ref','_backup')).tail(20)
                difference=(overlap.pct_chg_ref-overlap.pct_chg_backup).abs()
                if len(overlap)<3 or not np.isfinite(difference).all() or difference.max()>RETURN_TOLERANCE:
                    reason=f'adjustment check failed: overlap={len(overlap)}, max_return_difference={difference.max()}'
                    events.append({'symbol':symbol,'provider':provider,'error':reason})
                    print(f'[quote] {symbol} {provider} {reason}',flush=True)
                    continue
            combined=pd.concat([old,fresh],ignore_index=True).drop_duplicates('date',keep='last').sort_values('date')
            updates.append(fresh[COLUMNS])
            subset=eligible(combined)
            if len(subset)>len(best): best,best_source=subset,provider
            if len(subset)==len(required): return symbol,subset,provider,events,updates
            events.append({'symbol':symbol,'provider':provider,'error':f'incomplete dates/fields: {len(subset)}/{len(required)}'})
        return symbol,best,best_source,events,updates

    with ThreadPoolExecutor(max_workers=max(1,min(workers,3))) as pool:
        for symbol,frame,provider,events,updates in pool.map(one,sorted(set(symbols))):
            if len(frame)<len(required): failed.append(symbol)
            if not frame.empty: results.append(frame)
            if provider: sources[symbol]=provider
            errors.extend(events); writes.extend(updates)
    if cache_path and writes:
        combined=pd.concat([cached,*writes],ignore_index=True)
        combined=combined.drop_duplicates(['symbol','provider','adjust','date'],keep='last')
        combined=combined.sort_values(['symbol','provider','adjust','date']).groupby(['symbol','provider','adjust'],group_keys=False).tail(600)
        target=Path(cache_path); target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.parquet.tmp'); combined.to_parquet(temp,index=False); os.replace(temp,target)
    out=pd.concat(results,ignore_index=True) if results else pd.DataFrame(columns=COLUMNS)
    out.attrs.update(failed_symbols=failed,sources=sources,errors=errors)
    print(f'[quotes] symbols={len(set(symbols))} incomplete={len(failed)} sources={pd.Series(list(sources.values()),dtype=str).value_counts().to_dict()}',flush=True)
    if failed and not allow_partial: raise RuntimeError(f'历史行情下载不完整：{len(failed)} 只失败 {failed[:10]}')
    return out
