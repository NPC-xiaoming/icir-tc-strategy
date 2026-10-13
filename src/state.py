"""One atomic JSON ledger; parquet files are export mirrors."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class State:
    TABLES = ('holdings', 'ic_history', 'signal_cache', 'nav_history')

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = {}
        self.holdings = pd.DataFrame(columns=['symbol', 'name', 'weight', 'industry', 'benchmark_weight'])
        self.ic_history = pd.DataFrame()
        self.signal_cache = pd.DataFrame()
        self.nav_history = pd.DataFrame(columns=['date', 'nav', 'benchmark_nav'])

    @classmethod
    def load(cls, root: Path) -> 'State':
        state = cls(root)
        book = state.root / 'book.json'
        if not book.exists():
            legacy = [state.root / (n + '.parquet') for n in cls.TABLES]
            if (state.root / 'meta.json').exists() or any(p.exists() for p in legacy):
                raise ValueError('发现旧版或不完整账本。请备份 state 后人工核对迁移，不会自动清空或沿用错误净值。')
            return state
        payload = json.loads(book.read_text(encoding='utf-8'))
        if payload.get('schema_version') != 2:
            raise ValueError('Unsupported ledger schema')
        state.meta = payload['meta']
        for name in cls.TABLES:
            content = payload[name]
            frame = pd.DataFrame(content['rows'], columns=content['columns'])
            for column in ('date', 'effective_date'):
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column])
            setattr(state, name, frame)
        state.validate()
        return state

    def validate(self) -> None:
        if not self.holdings.empty:
            w = pd.to_numeric(self.holdings['weight'], errors='coerce')
            if not np.isfinite(w).all() or (w < -1e-10).any() or not np.isclose(w.sum(), 1., atol=1e-6):
                raise ValueError('Invalid holdings weights: must be finite, nonnegative and sum to one')
            if self.holdings.symbol.duplicated().any():
                raise ValueError('Duplicate holding symbols')
        if bool(self.meta.get('initialized')) != (not self.holdings.empty):
            raise ValueError('Ledger initialized flag and holdings disagree')
        if not self.nav_history.empty:
            values = self.nav_history[['nav', 'benchmark_nav']].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError('Invalid NAV in ledger')
            dates = pd.to_datetime(self.nav_history.date)
            if dates.duplicated().any() or not dates.is_monotonic_increasing:
                raise ValueError('Invalid NAV dates')
            if self.meta.get('last_run') != dates.iloc[-1].date().isoformat():
                raise ValueError('last_run must match final NAV date')

    def save(self) -> None:
        self.validate()
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {'schema_version': 2, 'meta': self.meta}
        for name in self.TABLES:
            frame = getattr(self, name)
            payload[name] = {'columns': list(frame.columns),
                             'rows': json.loads(frame.to_json(orient='records', date_format='iso'))}
        # Only book.json is authoritative. A failed mirror write never advances it.
        for name in self.TABLES:
            path = self.root / (name + '.parquet')
            temporary = path.with_suffix('.parquet.tmp')
            getattr(self, name).to_parquet(temporary, index=False)
            os.replace(temporary, path)
        atomic_text(self.root / 'meta.json', json.dumps(self.meta, ensure_ascii=False, indent=2, allow_nan=False))
        atomic_text(self.root / 'book.json', json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))

    @property
    def initialized(self) -> bool:
        return bool(self.meta.get('initialized', False)) and not self.holdings.empty

    @property
    def last_rebalance(self):
        return pd.Timestamp(self.meta['last_rebalance']) if self.meta.get('last_rebalance') else None

    @property
    def last_signal_date(self):
        return pd.Timestamp(self.meta['last_signal_date']) if self.meta.get('last_signal_date') else None

    @property
    def last_run(self):
        return pd.Timestamp(self.meta['last_run']) if self.meta.get('last_run') else None

    def weights(self) -> pd.Series:
        return self.holdings.set_index('symbol')['weight'].astype(float) if not self.holdings.empty else pd.Series(dtype=float)

    def append_nav(self, date, nav, benchmark_nav) -> None:
        date = pd.Timestamp(date)
        if not self.nav_history.empty and date <= pd.Timestamp(self.nav_history.iloc[-1]['date']):
            raise ValueError('NAV dates must advance strictly; duplicate application refused')
        row = pd.DataFrame([{'date': date, 'nav': float(nav), 'benchmark_nav': float(benchmark_nav)}])
        self.nav_history = row if self.nav_history.empty else pd.concat([self.nav_history, row], ignore_index=True)
        self.meta['last_run'] = date.date().isoformat()
