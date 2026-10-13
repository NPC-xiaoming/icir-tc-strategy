"""Risk model and the tracking-error-penalised optimiser (kappa = 100).

    max  w'r  -  (kappa/2)*ANNUAL*(w - w_b)'Sigma(w - w_b)  -  trading cost

subject to: full investment, single-name cap, minimum holding, absolute
deviation from benchmark, industry and size exposure caps, and a turnover
budget. Everything except the return input mirrors the research version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp


def estimate_covariance(
    returns_wide: pd.DataFrame,
    window: int = 250,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """Ledoit-Wolf shrunk daily covariance for the given symbols."""
    from sklearn.covariance import LedoitWolf

    data = returns_wide.tail(window).dropna(axis=1, how="all")
    if len(data) < window or data.empty or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError(f'Risk model requires {window} complete dated observations')
    symbols = list(data.columns)
    values = data.to_numpy(dtype=float)
    values = values - np.nanmean(values, axis=0, keepdims=True)
    values = np.nan_to_num(values, nan=0.0)
    cov = LedoitWolf().fit(values).covariance_
    cov = cov + ridge * np.eye(cov.shape[0])
    return pd.DataFrame(cov, index=symbols, columns=symbols)


def optimize(
    score: pd.Series,
    benchmark: pd.Series,
    covariance: pd.DataFrame,
    cfg: dict,
    pretrade: pd.Series | None = None,
    industry: pd.Series | None = None,
    size_z: pd.Series | None = None,
    annual: int = 242,
) -> dict:
    """Solve the TE-penalised optimisation; returns weights plus diagnostics."""
    opt = cfg["optimizer"]
    cost = cfg["cost"]
    if size_z is None:
        raise ValueError('size_z is required: configured size constraint cannot be skipped')

    symbols = list(benchmark.index)
    wb = pd.to_numeric(benchmark.reindex(symbols), errors="coerce")
    if (len(symbols) == 0 or not np.isfinite(wb).all() or (wb<=0).any()
            or not np.isfinite(score.reindex(symbols)).all()):
        raise ValueError('Incomplete optimizer inputs')
    wb = wb / wb.sum()
    r = pd.to_numeric(score.reindex(symbols), errors="coerce").fillna(0.0)
    exit_weight = 0.
    if pretrade is None:
        # First build: the book starts from cash, not from the benchmark, so
        # turnover is measured against an empty portfolio.
        w0 = pd.Series(0.0, index=symbols, dtype=float)
        is_first = True
    else:
        if not np.isfinite(pretrade).all() or (pretrade<0).any() or not np.isclose(pretrade.sum(),1.):
            raise ValueError('Invalid pretrade weights')
        exit_weight = float(pretrade.loc[~pretrade.index.isin(symbols)].sum())
        w0 = pd.to_numeric(pretrade.reindex(symbols), errors="coerce").fillna(0.0)
        is_first = False

    cov = covariance.reindex(index=symbols, columns=symbols).to_numpy(dtype=float)
    if not np.isfinite(cov).all():
        raise ValueError('Incomplete covariance matrix')
    cov = cov * annual
    m = len(symbols)

    kappa = float(opt["kappa"])
    max_weight = float(opt["max_weight"])
    min_position = float(opt["min_position"])
    deviation = float(opt["deviation"])

    lower = np.maximum(min_position, wb.to_numpy(dtype=float) - deviation)
    upper = np.minimum(max_weight, wb.to_numpy(dtype=float) + deviation)

    c_buy = (cost["commission_bps"] + cost["transfer_fee_bps"] + cost["slippage_bps"]) / 10_000.0
    c_sell = (
        cost["commission_bps"] + cost["stamp_duty_bps"] + cost["transfer_fee_bps"] + cost["slippage_bps"]
    ) / 10_000.0

    w = cp.Variable(m)
    up = cp.Variable(m)
    um = cp.Variable(m)
    te_term = (kappa / 2.0) * cp.quad_form(w - wb.to_numpy(dtype=float), cp.psd_wrap(cov))
    objective = cp.Maximize(
        r.to_numpy(dtype=float) @ w - te_term - c_buy * cp.sum(up) - c_sell * cp.sum(um)
    )
    constraints = [
        cp.sum(w) == 1.0,
        w >= lower,
        w <= upper,
        up >= w - w0.to_numpy(dtype=float),
        um >= w0.to_numpy(dtype=float) - w,
        up >= 0,
        um >= 0,
    ]
    if not is_first:
        constraints.append(cp.sum(up) + cp.sum(um) + exit_weight <= float(opt["turnover_theta"]))
    if industry is not None:
        labels = industry.reindex(symbols).fillna("其他").astype(str)
        theta = float(opt["industry_theta"])
        for _, group in labels.groupby(labels, sort=False):
            selector = np.zeros(m)
            for name in group.index:
                selector[symbols.index(name)] = 1.0
            bench_exposure = float(selector @ wb.to_numpy(dtype=float))
            constraints.append(selector @ w <= bench_exposure + theta)
            constraints.append(-selector @ w <= -bench_exposure + theta)
    if size_z is not None:
        z = pd.to_numeric(size_z.reindex(symbols), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(z).all():
            raise ValueError('Missing size exposure')
        theta = float(opt["size_theta"])
        bench_exposure = float(z @ wb.to_numpy(dtype=float))
        constraints.append(z @ w <= bench_exposure + theta)
        constraints.append(-z @ w <= -bench_exposure + theta)

    problem = cp.Problem(objective, constraints)
    status = None
    solver = str(opt.get('solver', 'OSQP'))
    if solver not in ('OSQP', 'CLARABEL'):
        raise ValueError(f'Unsupported solver: {solver}')
    try:
        if solver == 'OSQP':
            problem.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-7, eps_rel=1e-7)
        elif solver == 'CLARABEL':
            problem.solve(solver=cp.CLARABEL)
        else:
            raise ValueError(f'Unsupported solver: {solver}')
        status = problem.status
    except Exception:  # noqa: BLE001
        try:
            problem.solve(solver=cp.CLARABEL)
            status = problem.status
        except Exception:  # noqa: BLE001
            status = "failed"

    if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return {
            "weights": pretrade.copy() if pretrade is not None else pd.Series(dtype=float),
            "status": str(status),
            "tracking_error": float("nan"),
            "turnover": 0.0,
            "cost": 0.0,
            "fallback": True,
        }

    weights = pd.Series(np.clip(w.value, 0.0, None), index=symbols, dtype=float)
    weights = weights / weights.sum()
    violations = [float((lower-weights.to_numpy()).max()), float((weights.to_numpy()-upper).max())]
    if not is_first:
        violations.append(float((weights-w0).abs().sum())+exit_weight-float(opt['turnover_theta']))
    violations.append(abs(float((weights-wb)@pd.Series(z,index=symbols)))-float(opt['size_theta']))
    if industry is not None:
        violations.extend(((weights-wb).groupby(labels).sum().abs()-float(opt['industry_theta'])).tolist())
    if max(violations) > 1e-5 or not np.isfinite(weights).all():
        return {'weights': pretrade.copy() if pretrade is not None else pd.Series(dtype=float),
                'status': 'constraint_validation_failed', 'tracking_error': float('nan'),
                'turnover': 0., 'cost': 0., 'fallback': True}
    active = weights - wb
    te = float(np.sqrt(max(0.0, active.to_numpy() @ cov @ active.to_numpy())))
    turnover = float((weights - w0).abs().sum()) + exit_weight
    traded = (weights - w0)
    cost_est = float(
        traded.clip(lower=0.0).sum() * c_buy + ((-traded.clip(upper=0.0)).sum()+exit_weight) * c_sell
    )
    return {
        "weights": weights,
        "status": str(status),
        "tracking_error": te,
        "turnover": turnover,
        "cost": cost_est,
        "fallback": False,
    }
