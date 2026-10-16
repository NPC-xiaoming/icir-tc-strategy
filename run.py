"""After-close, idempotent paper portfolio runner. No brokerage execution."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from src import data as data_mod
from src import report as report_mod
from src.accounting import dated_returns, drift
from src.rebalance import build_plan
from src.state import State, atomic_text

ROOT = Path(__file__).resolve().parent


def load_config():
    cfg=json.loads((ROOT/'config/strategy.json').read_text(encoding='utf-8'))
    if cfg['index']['rebalance_rule']!='monthly_first_trading_day':
        raise ValueError('Only monthly_first_trading_day is supported')
    expected={'momentum_120_20':{'direction':1.,'lookback':120,'skip':20},
              'volatility_60':{'direction':-1.,'window':60},
              'turnover_20':{'direction':-1.,'window':20}}
    if cfg['factors']!=expected or cfg['annualization']!=242:
        raise ValueError('当前因子实现固定为 120/20、60、20 日及年化 242；更改规则需同步实现和测试')
    return cfg


def current_time():
    return pd.Timestamp.now(tz='Asia/Shanghai')


def trading_calendar():
    cache=ROOT/'state/market_calendar.parquet'
    try:
        raw=data_mod._retry(lambda:data_mod.ak.tool_trade_date_hist_sina(),'trading calendar')
        dates=pd.DatetimeIndex(pd.to_datetime(raw['trade_date'])).drop_duplicates().sort_values()
        if dates.empty: raise ValueError('Empty trading calendar')
        cache.parent.mkdir(parents=True,exist_ok=True)
        temp=cache.with_suffix('.parquet.tmp')
        pd.DataFrame({'date':dates}).to_parquet(temp,index=False); os.replace(temp,cache)
        return dates
    except Exception as exc:
        if not cache.exists(): raise
        print(f'[calendar] using cached exchange calendar: {exc}',flush=True)
        return pd.DatetimeIndex(pd.read_parquet(cache).date).drop_duplicates().sort_values()


def resolve_trade_date(calendar,today):
    eligible=calendar[calendar<=today]
    return pd.Timestamp(eligible.max()) if len(eligible) else None


def is_rebalance_day(calendar,trade_date):
    days=calendar[(calendar>=trade_date.replace(day=1))&(calendar<=trade_date)]
    return len(days)>0 and days.min()==trade_date


def run_daily():
    cfg=load_config()
    provider=cfg['data'].get('quote_provider','eastmoney')
    if provider not in ('eastmoney','sina'): raise ValueError('Unsupported quote provider')
    checkpoint_path=ROOT/'state/valuation_checkpoint.json'
    # A completed book is authoritative even if cleanup of an old checkpoint was interrupted.
    book_path=ROOT/'state/book.json'
    load_name='book.json'
    if checkpoint_path.is_file() and book_path.is_file():
        book_meta=json.loads(book_path.read_text(encoding='utf-8')).get('meta', {})
        load_name='valuation_checkpoint.json' if book_meta.get('provisional') else 'book.json'
    elif checkpoint_path.is_file():
        load_name='valuation_checkpoint.json'
    state=State.load(ROOT/'state', load_name)
    migrating=state.initialized and state.meta.get('quote_provider','eastmoney')!=provider
    replaying=checkpoint_path.is_file() and not migrating
    checkpoint_payload=None
    unresolved=[]
    now=current_time()
    today=now.tz_localize(None).normalize()
    calendar=trading_calendar()
    if calendar.empty or calendar.max()<today:
        raise ValueError('交易日历尚未覆盖今天，拒绝按旧日期运行')
    if today not in calendar:
        print('[skip] 非交易日，不重复计算上一交易日')
        return 0
    if now.hour<16:
        print('[skip] 北京时间 16:00 前不启动收盘跟踪，请收盘数据落定后再运行')
        return 0
    trade_date=resolve_trade_date(calendar,today)
    if migrating:
        if state.last_run is not None and state.last_run>trade_date:
            raise ValueError('Cannot migrate a future ledger')
        state.nav_history=state.nav_history.iloc[:0].copy()
        state.ic_history=pd.DataFrame()
        state.signal_cache=pd.DataFrame()
        state.meta={k:v for k,v in state.meta.items() if k in ('initialized','last_rebalance')}
        state.meta.update(quote_provider=provider,baseline_date=str(trade_date.date()),
                          migration_note='沿用旧口径最后可信持仓权重；切换日收盘建立新基准，不计算此前收益')
    if state.last_run is not None and state.last_run>=trade_date:
        print(f'[skip] {trade_date.date()} 已成功记账，不重复运行')
        return 0
    if os.getenv('ICIR_TC_FORCE_REBALANCE') or os.getenv('ICIR_TC_SYMBOL_LIMIT'):
        raise ValueError('已移除可污染真实账本的测试环境变量；请使用 tests/ 的临时数据测试')
    pending=calendar[(calendar>state.last_run)&(calendar<=trade_date)] if state.last_run is not None else pd.DatetimeIndex([trade_date])
    previous=calendar[calendar<pending[0]]
    if previous.empty:
        raise ValueError('缺少前一交易日，不能计算基准收益')
    signal_date=pd.Timestamp(calendar[calendar<trade_date].max())
    needs_rebalance=not migrating and (not state.initialized or state.last_rebalance is None
                     or state.last_rebalance.to_period('M')!=trade_date.to_period('M'))
    print(f'[run] through={trade_date.date()} pending={len(pending)} rebalance={needs_rebalance}',flush=True)
    held=state.weights().index.tolist()
    constituents=None
    symbols=set(held)
    if needs_rebalance:
        constituents=data_mod.fetch_constituents(cfg['index']['code'])
        if len(constituents)!=int(cfg['data'].get('expected_constituents',300)):
            raise ValueError('指数成分数量不完整，拒绝改变投资范围')
        symbols.update(constituents.symbol)
        if not state.signal_cache.empty:
            symbols.update(state.signal_cache.symbol)
    # Rebalance downloads warm history; daily valuation downloads only missing dates.
    start=None if needs_rebalance else pending[0]
    if needs_rebalance:
        warm_start=trade_date-pd.Timedelta(days=int(cfg['data']['history_days']*1.7))
        start=min(warm_start,pending[0],state.last_rebalance) if state.last_rebalance is not None else warm_start
    requested_symbols=sorted(symbols)
    history=data_mod.fetch_stock_history(requested_symbols,days=int(cfg['data']['history_days']),
        start_date=start,end_date=trade_date,workers=int(cfg['data'].get('workers',3)),
        timeout=float(cfg['data'].get('timeout_seconds',12)),allow_partial=True,
        cache_path=ROOT/'state/market_history.parquet',
        required_dates=calendar[(calendar>=start)&(calendar<=trade_date)],calendar=calendar,
        require_turnover=needs_rebalance,primary_provider=provider) if symbols else pd.DataFrame()
    quality={'through':str(trade_date.date()),'sources':history.attrs.get('sources',{}),
             'errors':history.attrs.get('errors',[]),'days':[]}
    atomic_text(ROOT/'logs/data_quality.json',json.dumps(quality,ensure_ascii=False,indent=2))
    missing=set(history.attrs.get('failed_symbols', []))
    missing.update(set(requested_symbols)-set(history.symbol.unique()))
    held_returns=dated_returns(history,pending,held,allow_missing=True)
    index=data_mod.fetch_index_history(cfg['index']['code'],start_date=previous[-1],end_date=trade_date,
        required_dates=pd.DatetimeIndex([previous[-1],*pending]),
        provider='sina' if provider=='sina' else None)
    if index.date.duplicated().any():
        raise ValueError('Duplicate official index dates')
    closes=index.set_index('date').close.reindex(pd.DatetimeIndex([previous[-1],*pending])).astype(float)
    if not np.isfinite(closes).all() or (closes<=0).any():
        raise ValueError('官方基准行情缺失或过期，未提交状态')
    index_returns=closes.pct_change(fill_method=None).reindex(pending)
    outputs={}
    cold_note='历史 IC 未充分预热时使用等权因子回退；不代表已复现研究回测。'
    for day in pending:
        current=history.loc[history.date==day].set_index('symbol')['pct_chg'].reindex(held)
        current=pd.to_numeric(current,errors='coerce')
        bad=~np.isfinite(current)|(current<=-1)
        day_missing=current.index[bad].tolist()
        missing_weight=float(state.weights().reindex(day_missing,fill_value=0.).sum())
        quality['days'].append({'date':str(day.date()),'held_missing':day_missing,'missing_weight':missing_weight})
        atomic_text(ROOT/'logs/data_quality.json',json.dumps(quality,ensure_ascii=False,indent=2))
        limit=float(cfg['data'].get('max_missing_weight',.05))
        if migrating and day_missing:
            raise ValueError('建立新浪基准日需要全部持仓行情，未提交迁移')
        if held and (len(day_missing)==len(held) or missing_weight>limit):
            raise ValueError(f'{day.date()} 持仓行情缺失 {len(day_missing)}/{len(held)} 只、权重 {missing_weight:.2%}，超过 {limit:.0%} 保护阈值；未提交账本，缓存保留供重试')
        if day_missing:
            if checkpoint_payload is None:
                checkpoint_payload=state.payload()
            unresolved.extend({'date':str(day.date()),'symbol':symbol} for symbol in day_missing)
        had_nav=not state.nav_history.empty
        old_nav=float(state.nav_history.iloc[-1]['nav']) if had_nav else 1.
        old_bench=float(state.nav_history.iloc[-1]['benchmark_nav']) if had_nav else 1.
        gross,pretrade=drift(state.weights(),held_returns.loc[day])
        if migrating:
            gross=0.; pretrade=state.weights()
        if state.initialized:
            state.holdings['weight']=state.holdings.symbol.map(pretrade)
        bench_ret=float(index_returns.loc[day]) if had_nav else 0.
        fee=0.; actual_rebalance=False; notes=[]
        if unresolved and needs_rebalance:
            notes.append('持仓行情尚有缺失，暂缓调仓，补齐估值后重试。')
        if replaying:
            notes.append('已从暂估前检查点重新计算，本报告替代此前暂估结果。')
        if len(pending)>1:
            notes.append('缺日按原持仓补估值；未倒填过去调仓。')
        if day==trade_date and needs_rebalance and not unresolved:
            available=set(history.symbol.unique())
            constituents=constituents[constituents.symbol.isin(available)].copy()
            if constituents.empty:
                raise ValueError('当前没有可用成分股，等待下次行情补齐')
            constituents['weight']=constituents['weight']/constituents['weight'].sum()
            missing.update(set(requested_symbols)-available)
            required_dates=calendar[calendar<=signal_date][-max(int(cfg['risk_model']['window']),121):]
            valid_symbols=set()
            for symbol in constituents.symbol:
                rows=history[history.symbol==symbol].set_index('date').reindex(required_dates)
                if not rows[['pct_chg','close','turnover']].isna().any().any() and (rows[['pct_chg','close','turnover']].to_numpy(dtype=float)>-1).all():
                    valid_symbols.add(symbol)
            missing.update(set(constituents.symbol)-valid_symbols)
            constituents=constituents[constituents.symbol.isin(valid_symbols)].copy()
            if constituents.empty:
                raise ValueError('没有成分股满足完整因子历史窗口，等待后续行情补齐')
            constituents['weight']=constituents['weight']/constituents['weight'].sum()
            industry=data_mod.load_industry(ROOT/'state/industry_sw_l1.parquet')
            unadjusted=data_mod.fetch_stock_history(constituents.symbol.tolist(),adjust='',
                start_date=day,end_date=day,workers=int(cfg['data'].get('workers',3)),
                timeout=float(cfg['data'].get('timeout_seconds',12)),allow_partial=True,
                cache_path=ROOT/'state/market_history.parquet',required_dates=pd.DatetimeIndex([day]),
                calendar=calendar,primary_provider=provider)
            spot=(data_mod.fetch_spot(history=unadjusted) if provider=='sina' else data_mod.fetch_spot())
            spot_symbols=set(spot.loc[spot['close'].notna() & spot['circ_mv'].notna(),'symbol'])
            missing.update(set(constituents.symbol)-spot_symbols)
            constituents=constituents[constituents.symbol.isin(spot_symbols)].copy()
            if constituents.empty:
                raise ValueError('当前没有可用收盘/流通市值数据，等待下次行情补齐')
            constituents['weight']=constituents['weight']/constituents['weight'].sum()
            missing.update(unadjusted.attrs.get('failed_symbols', []))
            constituents=constituents[constituents.symbol.isin(set(unadjusted.symbol.unique()))].copy()
            constituents['weight']=constituents['weight']/constituents['weight'].sum()
            holdings,changes,diag,fw=build_plan(state,history,constituents,industry,spot,cfg,day,signal_date,calendar,
                                              unadjusted_history=unadjusted)
            actual_rebalance=not diag['fallback']
            if actual_rebalance:
                fee=float(diag['cost']); state.holdings=holdings
            else:
                notes.append('优化失败，保留原持仓（初始组合保留现金），下个交易日重试。')
            if not is_rebalance_day(calendar,day):
                notes.append('本次为首次建仓或延期调仓，执行日为本日。')
            if diag['cold_start']: notes.append(cold_note)
            text=report_mod.render_rebalance(day,signal_date,fw,state.holdings,changes,diag,cfg)
            text+='\n'+ '\n'.join('- '+n for n in notes)+'\n'
            outputs[ROOT/'reports/rebalance'/f'{day.date()}.md']=text
            if not changes.empty:
                outputs[ROOT/'reports/rebalance'/f'{day.date()}_trades.csv']=changes.to_csv(index=False)
            state.meta['last_status']=diag['status']
        nav=old_nav*(1+gross)*(1-fee)
        net=(1+gross)*(1-fee)-1
        benchmark_nav=old_bench*(1+bench_ret)
        state.append_nav(day,nav,benchmark_nav)
        state.meta.update(nav=nav,benchmark_nav=benchmark_nav)
        text=report_mod.render_daily(day,nav,benchmark_nav,net,bench_ret,nav/benchmark_nav-1,
                                     state.holdings,actual_rebalance,state.nav_history)
        text+=f'\n- 本日交易费用（净资产比例）：{fee:.6%}\n- 版本：{cfg["version"]}\n'
        text+='- 基准：沪深300官方价格指数；组合为含固定费用的权重型研究模拟。\n'
        text+=f'- 行情口径：{provider} 后复权收盘价日收益；模拟净值不代表真实资金盈亏。\n'
        if state.meta.get('baseline_date'):
            text+=f"- 新口径基准日：{state.meta['baseline_date']}，组合和沪深300均从 1 起算；此前旧口径报告不参与累计。\n"
        if migrating:
            text+='- 本日仅建立新基准，沿用最后可信持仓权重，不计算切换前收益，不发生调仓；收益跟踪从下一交易日开始。\n'
        text+=f'- 当日持仓行情缺失：{", ".join(day_missing) if day_missing else "无"}；缺失权重 {missing_weight:.2%}。\n'
        if actual_rebalance:
            text+=f'- 调仓数据不完整或被过滤的股票：{", ".join(sorted(missing)) if missing else "无"}。\n'
        text+=f'- 数据来源：{json.dumps(pd.Series(list(quality['sources'].values()),dtype=str).value_counts().to_dict(),ensure_ascii=False)}\n' if quality['sources'] else ''
        if unresolved:
            text='> **暂估报告：存在未补齐持仓行情，收益和超额暂不可作为有效绩效；下次从检查点重新计算。**\n\n'+text
        text+='- 因子使用前一交易日收盘数据，风险约束和成交采用执行日可用信息；收盘模拟并非可执行成交承诺。\n'
        text+='- '+cold_note+'\n'+'\n'.join('- '+n for n in notes if n!=cold_note)+'\n'
        outputs[ROOT/'reports/daily'/f'{day.date()}.md']=text
    state.meta.update(nav=nav,benchmark_nav=benchmark_nav,version=cfg['version'],quote_provider=provider,
        config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        recorded_at=now.isoformat(),benchmark='沪深300官方价格指数',
        missing_symbols=sorted(missing),held_missing_symbols=sorted({x['symbol'] for x in unresolved}),
        provisional=bool(unresolved),unresolved_quotes=unresolved)
    metrics=report_mod.make_nav_plot(state.nav_history,ROOT/'reports/nav_curve.png',
                                     ROOT/'state/backtest_nav_history.csv')
    state.meta['chart_metrics']=metrics
    state.validate()
    if migrating:
        archive=ROOT/'state/archive'/f'pre-{provider}'
        if not archive.exists():
            archive.mkdir(parents=True)
            for name in ('book.json','valuation_checkpoint.json'):
                source=ROOT/'state'/name
                if source.is_file(): shutil.copy2(source,archive/name)
        report_archive=ROOT/'reports/archive'/f'pre-{provider}'
        if not report_archive.exists():
            report_archive.mkdir(parents=True)
            for name in ('daily','rebalance'):
                source=ROOT/'reports'/name
                if source.is_dir(): shutil.copytree(source,report_archive/name)
    for path,text in outputs.items(): atomic_text(path,text)
    # Save the replay checkpoint first; interruption may cause replay, never lost corrections.
    if checkpoint_payload is not None:
        atomic_text(checkpoint_path,json.dumps(checkpoint_payload,ensure_ascii=False,indent=2))
    state.save()
    if not unresolved and checkpoint_path.exists(): checkpoint_path.unlink()
    print(f'[done] nav={nav:.6f} benchmark={benchmark_nav:.6f}; reports and ledger saved',flush=True)
    return 0


def main():
    with data_mod.network_timeout(12):
        return run_daily()


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
