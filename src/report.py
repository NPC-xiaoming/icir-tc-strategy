"""Markdown report generation for the daily run and the monthly rebalance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Use a CJK font when available on the runner or local workstation.
_fonts={font.name for font in font_manager.fontManager.ttflist}
for _name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "PingFang HK", "STHeiti"):
    if _name in _fonts:
        plt.rcParams["font.sans-serif"]=[_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"]=False


def _pct(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def _table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    if frame.empty:
        return "_（无数据）_\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def render_daily(
    trade_date: pd.Timestamp,
    nav: float,
    benchmark_nav: float,
    daily_return: float,
    benchmark_return: float,
    cumulative_excess: float,
    holdings: pd.DataFrame,
    is_rebalance: bool,
    lookback_nav: pd.DataFrame,
) -> str:
    """Daily monitoring note."""
    recent = lookback_nav.tail(20)
    lines = [
        f"# ICIR-TC 每日运行结果 · {trade_date.date()}",
        "",
        f"- 组合净值：**{nav:.4f}**（当日 {_pct(daily_return)}）",
        f"- 基准净值：{benchmark_nav:.4f}（当日 {_pct(benchmark_return)}）",
        f"- 累计超额：**{_pct(cumulative_excess)}**",
        f"- 今日操作：{'**调仓日**（已生成调仓报告）' if is_rebalance else '无调仓（权重自然漂移）'}",
        "",
        "## 当前持仓摘要",
        "",
    ]
    top = holdings.sort_values("weight", ascending=False).head(10).copy()
    top["weight_pct"] = top["weight"].map(lambda v: _pct(v))
    lines.append(
        _table(
            top,
            ["symbol", "name", "weight_pct", "industry"],
            ["代码", "名称", "权重", "行业"],
        )
    )
    lines += [
        f"持仓股票数：{len(holdings)}",
        "",
        "## 近 20 个交易日净值",
        "",
    ]
    if not recent.empty:
        tail = recent.copy()
        tail["date"] = pd.to_datetime(tail["date"]).dt.date.astype(str)
        tail["nav_f"] = tail["nav"].map(lambda v: f"{v:.4f}")
        tail["bench_f"] = tail["benchmark_nav"].map(lambda v: f"{v:.4f}")
        lines.append(_table(tail, ["date", "nav_f", "bench_f"], ["日期", "组合净值", "基准净值"]))
    return "\n".join(lines) + "\n"


def make_nav_plot(nav_history: pd.DataFrame, output: Path, backtest_path: Path | None = None) -> dict:
    """Save one comparison chart and return the latest derived metrics."""
    frame=nav_history.copy()
    frame['date']=pd.to_datetime(frame['date'])
    frame=frame.sort_values('date').drop_duplicates('date', keep='last')
    online=frame['nav']/float(frame['nav'].iloc[0])
    benchmark=frame['benchmark_nav']/float(frame['benchmark_nav'].iloc[0])
    excess=online/benchmark-1
    drawdown=online/online.cummax()-1
    fig, axes=plt.subplots(3,1,figsize=(12,10),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
    axes[0].plot(frame.date,online,label='线上组合（首日=1）',marker='o' if len(frame)==1 else None)
    axes[0].plot(frame.date,benchmark,label='沪深300（线上首日=1）',marker='o' if len(frame)==1 else None)
    if backtest_path and backtest_path.is_file():
        bt=pd.read_csv(backtest_path)
        required={'date','nav','benchmark_nav'}
        if required.issubset(bt.columns):
            bt['date']=pd.to_datetime(bt['date'], errors='coerce')
            bt['nav']=pd.to_numeric(bt['nav'], errors='coerce')
            bt['benchmark_nav']=pd.to_numeric(bt['benchmark_nav'], errors='coerce')
            bt=bt.dropna(subset=list(required)).sort_values('date').drop_duplicates('date', keep='last')
            if not bt.empty and (bt[['nav','benchmark_nav']] > 0).all().all():
                bt_nav=bt['nav']/float(bt['nav'].iloc[0])
                bt_benchmark=bt['benchmark_nav']/float(bt['benchmark_nav'].iloc[0])
                bt_excess=bt_nav/bt_benchmark-1
                bt_drawdown=bt_nav/bt_nav.cummax()-1
                axes[0].plot(bt.date,bt_nav,label='本地回测组合（首日=1）',alpha=.8)
                axes[0].plot(bt.date,bt_benchmark,label='本地回测沪深300（首日=1）',alpha=.8)
                axes[1].plot(bt.date,bt_excess,label='本地回测超额',alpha=.8)
                axes[2].plot(bt.date,bt_drawdown,label='本地回测回撤',alpha=.8)
    axes[0].set_ylabel('净值'); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(frame.date,excess,label='线上超额',marker='o' if len(frame)==1 else None)
    axes[1].axhline(0,color='black',lw=.7)
    axes[1].set_ylabel('超额'); axes[1].legend(); axes[1].grid(alpha=.25)
    axes[2].plot(frame.date,drawdown,label='线上回撤',marker='o' if len(frame)==1 else None)
    axes[2].fill_between(frame.date,drawdown,0,alpha=.25)
    axes[2].set_ylabel('回撤'); axes[2].legend(); axes[2].grid(alpha=.25)
    fig.suptitle('ICIR-TC 模拟组合净值跟踪'); fig.tight_layout(); output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150); plt.close(fig)
    return {'online_nav':float(online.iloc[-1]),'benchmark_nav':float(benchmark.iloc[-1]),
            'excess':float(excess.iloc[-1]),'drawdown':float(drawdown.iloc[-1])}


def render_rebalance(
    effective_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    factor_weights: dict[str, float],
    holdings: pd.DataFrame,
    changes: pd.DataFrame,
    diagnostics: dict,
    config: dict,
) -> str:
    """Monthly rebalance note with the full target book and the trade list."""
    opt = config["optimizer"]
    lines = [
        f"# ICIR-TC 调仓报告 · {effective_date.date()}",
        "",
        f"- 信号日：{signal_date.date()}",
        f"- 生效日：**{effective_date.date()}**",
        f"- 优化器：TE 惩罚 QP，κ = {opt['kappa']:g}",
        "",
        "## 一、因子权重（滚动 IC-IR）",
        "",
        "| 因子 | 权重 |",
        "|---|---|",
    ]
    for factor, weight in factor_weights.items():
        lines.append(f"| {factor} | {weight:.3f} |")
    lines += [
        "",
        "## 二、优化器诊断",
        "",
        "| 项目 | 数值 |",
        "|---|---|",
        f"| 求解状态 | {diagnostics.get('status', '—')} |",
        f"| 预估跟踪误差（年化） | {_pct(diagnostics.get('tracking_error'))} |",
        f"| 双边换手（预算 {opt['turnover_theta']:g}） | {_pct(diagnostics.get('turnover'))} |",
        f"| 预估交易成本 | {_pct(diagnostics.get('cost'), 3)} |",
        f"| 是否降级 | {'是' if diagnostics.get('fallback') else '否'} |",
        f"| 处理结果 | {'保留原持仓；首次运行保持现金' if diagnostics.get('fallback') else '更新模拟持仓并扣费'} |",
        f"| 已积累 IC / 最早预热门槛 | {diagnostics.get('ic_observations', 0)} / {diagnostics.get('ic_warmup_required', '—')} |",
        f"| 指数权重快照 | {diagnostics.get('weight_snapshot', '—')} |",
        "",
        "## 三、交易清单",
        "",
    ]
    if changes.empty:
        lines.append("_本次无交易_\n")
    else:
        trades = changes.loc[changes["trade_weight"].abs() > 1e-8].copy()
        trades["dir"] = trades["trade_weight"].map(lambda v: "买入" if v > 0 else "卖出")
        trades["w_before"] = trades["pretrade_weight"].map(lambda v: _pct(v))
        trades["w_after"] = trades["target_weight"].map(lambda v: _pct(v))
        trades["delta"] = trades["trade_weight"].map(lambda v: f"{v * 100:+.2f}%")
        trades = trades.reindex(trades["trade_weight"].abs().sort_values(ascending=False).index)
        lines.append(
            _table(
                trades.head(40),
                ["symbol", "name", "dir", "w_before", "w_after", "delta"],
                ["代码", "名称", "方向", "调仓前", "调仓后", "变动"],
            )
        )
        lines.append(f"共 {len(trades)} 笔交易（上表按变动幅度显示前 40 笔）\n")
    lines += [
        "",
        "## 四、目标持仓（前 30 大）",
        "",
    ]
    top = holdings.sort_values("weight", ascending=False).head(30).copy()
    top["weight_pct"] = top["weight"].map(lambda v: _pct(v))
    top["bench_pct"] = top["benchmark_weight"].map(lambda v: _pct(v))
    top["active_pct"] = (top["weight"] - top["benchmark_weight"]).map(lambda v: f"{v * 100:+.2f}%")
    lines.append(
        _table(
            top,
            ["symbol", "name", "weight_pct", "bench_pct", "active_pct", "industry"],
            ["代码", "名称", "目标权重", "基准权重", "主动偏离", "行业"],
        )
    )
    lines += [
        f"目标持仓股票数：{len(holdings)}",
        "",
        "## 五、约束核对",
        "",
        f"- 单股上限 {_pct(opt['max_weight'], 0)}：最大持仓 {_pct(holdings['weight'].max())}",
        f"- 最小持仓 {_pct(opt['min_position'], 2)}：最小持仓 {_pct(holdings['weight'].min())}",
        f"- 个股偏离 ±{_pct(opt['deviation'], 1)}：实际最大偏离 "
        f"{_pct((holdings['weight'] - holdings['benchmark_weight']).abs().max())}",
        f"- 行业暴露 ±{_pct(opt['industry_theta'], 0)}（按申万一级分组核对）",
        f"- 实际最大行业主动暴露：{_pct(diagnostics.get('industry_max_active'))}",
        f"- 市值主动暴露上限 ±{opt['size_theta']}（对数流通市值标准分）：{diagnostics.get('size_active', '—')}",
        "- 首次从现金建仓不受换手预算限制；后续双边换手包括退出指数股票的卖出。",
        "- 优化失败时本页持仓为保留组合，约束可能超限，不视作成功调仓。",
        "",
        "---",
        "",
        "_本报告由 ICIR-TC 每日云端任务自动生成，仅为策略跟踪记录，不构成投资建议。_",
    ]
    return "\n".join(lines) + "\n"
