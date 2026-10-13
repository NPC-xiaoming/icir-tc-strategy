"""Prepare one paper rebalance from prior-session factors and dated inputs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import signal as signals
from .accounting import dated_returns, trading_cost
from .optimizer import estimate_covariance, optimize


def build_plan(state, history, constituents, industry, spot, cfg, trade_date, signal_date, calendar,
               unadjusted_history):
    symbols = constituents.symbol.tolist()
    benchmark = constituents.set_index('symbol').weight.astype(float)
    if not industry.reindex(symbols).notna().all():
        raise ValueError('行业缓存缺少当前成分，请刷新行业缓存后重试')
    snapshot = pd.Timestamp(constituents.as_of.max())
    if snapshot > signal_date or (trade_date-snapshot).days > cfg['data'].get('max_weight_age_days', 45):
        raise ValueError('指数权重快照时间无效或过期，拒绝调仓')
    # Spot uses unadjusted returns; HFQ returns are a different price series.
    current = dated_returns(unadjusted_history, [trade_date], symbols).iloc[0]
    spot = spot.drop_duplicates('symbol').set_index('symbol').reindex(symbols)
    caps = pd.to_numeric(spot['circ_mv'], errors='coerce')
    observed = pd.to_numeric(spot['pct_chg'], errors='coerce')
    if not np.isfinite(caps).all() or (caps <= 0).any():
        raise ValueError('流通市值数据不完整，不能省略市值约束')
    if not np.isfinite(observed).all() or (observed-current).abs().max() > .00011:
        raise ValueError('全市场快照与目标日收盘行情不一致，拒绝使用未确认日期的市值')
    size = np.log(caps)
    size = (size-size.mean())/size.std(ddof=0) if size.std(ddof=0)>1e-12 else size*0.

    # Preserve existing rolling-IC rules; never manufacture a warm history.
    if state.last_rebalance is not None and not state.signal_cache.empty:
        cached = state.signal_cache.set_index('symbol')[list(signals.FACTOR_NAMES)]
        held_dates = calendar[(calendar > state.last_rebalance)&(calendar <= signal_date)]
        dated_returns(history, held_dates, cached.index)
        realized = signals.forward_return(history, state.last_rebalance, signal_date)
        ic = signals.compute_ic(cached.astype(float), realized)
        if any(np.isfinite(v) for v in ic.values()):
            row = pd.DataFrame([{'effective_date': state.last_rebalance, **ic}])
            state.ic_history = (row if state.ic_history.empty else pd.concat([state.ic_history,row],ignore_index=True))
            state.ic_history = state.ic_history.drop_duplicates('effective_date',keep='last').sort_values('effective_date')
    window = int(cfg['risk_model']['window'])
    required_dates = calendar[calendar <= signal_date][-max(window,121):]
    if len(required_dates)<max(window,121):
        raise ValueError('交易日历不足以验证因子和风险窗口')
    wide = dated_returns(history,required_dates,symbols)
    for column,dates in [('close',required_dates[-121:]),('turnover',required_dates[-20:])]:
        values=history.pivot(index='date',columns='symbol',values=column).reindex(index=dates,columns=symbols).to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values<0).any() or (column=='close' and (values==0).any()):
            raise ValueError(f'因子历史 {column} 缺失或无效，拒绝静默跳过观测')
    raw = signals.compute_raw_factors(history, signal_date).reindex(symbols)
    if not np.isfinite(raw[list(signals.FACTOR_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError('当前成分的因子历史不完整，拒绝缩小股票池或重新归一化基准')
    z = signals.neutral_zscore(raw,industry,min_industry_size=int(cfg['signal']['min_industry_size']))
    factor_weights = signals.icir_weights(state.ic_history,window=int(cfg['signal']['ic_window']),lag=int(cfg['signal']['ic_lag']))
    score = signals.composite(z,factor_weights)
    covariance = estimate_covariance(wide,window=window,ridge=float(cfg['risk_model']['ridge']))
    pretrade = state.weights()
    result = optimize(score,benchmark,covariance,cfg,pretrade=pretrade if state.initialized else None,
                      industry=industry,size_z=size,annual=int(cfg['annualization']))
    diagnostics = {k:v for k,v in result.items() if k!='weights'}
    diagnostics['ic_observations'] = len(state.ic_history)
    diagnostics['ic_warmup_required'] = max(3,int(cfg['signal']['ic_window'])//2)+int(cfg['signal']['ic_lag'])
    diagnostics['cold_start'] = len(state.ic_history)<diagnostics['ic_warmup_required']
    diagnostics['weight_snapshot'] = snapshot.date().isoformat()
    if result['fallback']:
        return state.holdings.copy(),pd.DataFrame(),diagnostics,factor_weights
    target = result['weights']
    fee,turnover=trading_cost(pretrade,target,cfg['cost'])
    diagnostics.update(cost=fee,turnover=turnover,size_active=float((target-benchmark)@size))
    diagnostics['industry_max_active'] = float((target-benchmark).groupby(industry.reindex(symbols)).sum().abs().max())
    ids=pretrade.index.union(target.index)
    names=constituents.set_index('symbol')['name'].to_dict()
    if not state.holdings.empty:
        names={**state.holdings.set_index('symbol').name.to_dict(),**names}
    changes=pd.DataFrame({'symbol':ids,'name':[names.get(s,s) for s in ids],
                          'pretrade_weight':pretrade.reindex(ids,fill_value=0.).values,
                          'target_weight':target.reindex(ids,fill_value=0.).values})
    changes['trade_weight']=changes.target_weight-changes.pretrade_weight
    holdings=pd.DataFrame({'symbol':target.index,'name':[names.get(s,s) for s in target.index],
        'weight':target.values,'benchmark_weight':benchmark.reindex(target.index).values,
        'industry':industry.reindex(target.index).values})
    state.signal_cache=z.rename_axis('symbol').reset_index()
    state.meta.update(initialized=True,last_rebalance=trade_date.date().isoformat(),
                      last_signal_date=signal_date.date().isoformat(),factor_weights=factor_weights)
    return holdings,changes,diagnostics,factor_weights
