"""Network-free behavioral regression tests for the real runner and ledger."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run as app
from src.state import State

PROJECT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'config').mkdir()
        cfg = json.loads((PROJECT / 'config/strategy.json').read_text())
        cfg['data']['expected_constituents'] = 12
        (self.root / 'config/strategy.json').write_text(json.dumps(cfg))
        self.symbols = [f'{i:06d}' for i in range(1, 13)]
        self.day = pd.Timestamp('2026-06-03')
        dates = pd.bdate_range(end=self.day, periods=340)
        frames = []
        for i, symbol in enumerate(self.symbols):
            ret = .001 * np.sin(np.arange(len(dates)) * (i+1) / 20)
            frames.append(pd.DataFrame({'date': dates, 'symbol': symbol,
                'pct_chg': ret, 'close': 10*np.cumprod(1+ret), 'turnover': .5+i/10}))
        self.history = pd.concat(frames, ignore_index=True)
        self.history.loc[self.history.date >= '2026-06-01', 'pct_chg'] = .01
        self.cons = pd.DataFrame({'symbol': self.symbols, 'name': self.symbols,
            'weight': 1/12, 'as_of': pd.Timestamp('2026-05-29')})
        self.spot = pd.DataFrame({'symbol': self.symbols, 'name': self.symbols,
            'pct_chg': .01, 'close': 10., 'turnover': 1.,
            'circ_mv': np.linspace(1e9, 2e9, 12)})
        self.calendar = pd.bdate_range('2024-01-01', '2026-12-31')

    def seed(self, last='2026-06-02', symbols=None, weights=None, rebalance='2026-06-01'):
        s = State.load(self.root/'state')
        syms = symbols or self.symbols
        s.holdings = pd.DataFrame({'symbol': syms, 'name': syms,
            'weight': weights or [1/len(syms)]*len(syms), 'industry': 'I',
            'benchmark_weight': [1/len(syms)]*len(syms)})
        s.meta.update(initialized=True, last_run=last, last_rebalance=rebalance)
        s.append_nav(pd.Timestamp(last), 1., 1.)
        s.save()
        return s

    def invoke(self, date=None):
        date = pd.Timestamp(date or self.day)
        def history(symbols, **kwargs):
            frame = self.history[self.history.symbol.isin(symbols)].copy()
            if kwargs.get('adjust') == '' and hasattr(self,'unadjusted_return'):
                frame['pct_chg']=self.unadjusted_return
            if kwargs.get('end_date') is not None:
                frame = frame[frame.date <= pd.Timestamp(kwargs['end_date'])]
            if kwargs.get('start_date') is not None:
                frame = frame[frame.date >= pd.Timestamp(kwargs['start_date'])]
            return frame
        index_dates = self.calendar[self.calendar <= date]
        index = pd.DataFrame({'date': index_dates, 'close': 100.})
        index.loc[index.date >= '2026-06-03', 'close'] = 102.
        with contextlib.ExitStack() as st:
            for obj, name, value in [
                (app, 'ROOT', self.root),
                (app, 'current_time', lambda: date.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=18)),
                (app, 'trading_calendar', lambda: self.calendar),
                (app, 'resolve_trade_date', lambda *_: date),
                (app.data_mod, 'fetch_constituents', lambda *_: self.cons.copy()),
                (app.data_mod, 'fetch_spot', lambda: self.spot.copy()),
                (app.data_mod, 'fetch_stock_history', history),
                (app.data_mod, 'fetch_index_history', lambda *a, **kw: index.copy()),
                (app.data_mod, 'load_industry', lambda *_: pd.Series('I', index=self.symbols)),
            ]:
                st.enter_context(patch.object(obj, name, value, create=True))
            with contextlib.redirect_stdout(io.StringIO()):
                return app.main()

    def test_duplicate_day_does_not_compound(self):
        self.seed()
        self.invoke(); self.invoke()
        self.assertAlmostEqual(State.load(self.root/'state').nav_history.iloc[-1]['nav'], 1.01)

    def test_weights_drift_after_prices_change(self):
        self.seed(symbols=self.symbols[:2], weights=[.5, .5])
        self.history.loc[(self.history.date == self.day) & (self.history.symbol == '000001'), 'pct_chg'] = .1
        self.history.loc[(self.history.date == self.day) & (self.history.symbol == '000002'), 'pct_chg'] = 0.
        self.spot.loc[self.spot.symbol == '000001', 'pct_chg'] = .1
        self.spot.loc[self.spot.symbol == '000002', 'pct_chg'] = 0.
        self.invoke()
        self.assertAlmostEqual(State.load(self.root/'state').weights()['000001'], .5238095238095238)

    def test_missing_days_are_revalued_in_sequence(self):
        self.seed(last='2026-06-01')
        self.invoke()
        s = State.load(self.root/'state')
        self.assertEqual(s.nav_history.date.dt.strftime('%Y-%m-%d').tolist(),
                         ['2026-06-01', '2026-06-02', '2026-06-03'])
        self.assertAlmostEqual(s.nav_history.iloc[-1]['nav'], 1.0201)
        self.assertTrue((self.root/'reports/daily/2026-06-02.md').exists())

    def test_missing_quote_is_recorded_and_ledger_advances(self):
        self.seed()
        self.history = self.history[self.history.date < self.day]
        self.spot['pct_chg'] = np.nan
        self.invoke()
        s=State.load(self.root/'state')
        self.assertEqual(s.meta['last_run'], '2026-06-03')
        self.assertTrue(s.meta['missing_symbols'])

    def test_removed_constituent_still_earns_its_return(self):
        self.seed(symbols=['000001','000002'], weights=[.5,.5])
        self.cons = self.cons[self.cons.symbol != '000002']
        self.cons['weight'] = 1/11
        self.history.loc[(self.history.date == self.day) & (self.history.symbol == '000002'), 'pct_chg'] = .1
        self.spot.loc[self.spot.symbol == '000002','pct_chg'] = .1
        self.invoke()
        self.assertAlmostEqual(State.load(self.root/'state').nav_history.iloc[-1]['nav'], 1.055)

    def test_midmonth_first_run_builds_book_and_deducts_cost(self):
        self.invoke()
        s = State.load(self.root/'state')
        self.assertTrue(s.initialized)
        self.assertAlmostEqual(s.nav_history.iloc[-1]['nav'], .99934, places=7)
        self.assertTrue((self.root/'reports/daily/2026-06-03.md').is_file())
        self.assertTrue((self.root/'reports/rebalance/2026-06-03_trades.csv').is_file())
        self.assertEqual(s.meta['last_signal_date'], '2026-06-02')

    def test_late_monthly_rebalance_is_not_backdated(self):
        self.seed(last='2026-05-29', rebalance='2026-05-04')
        self.invoke()
        s = State.load(self.root/'state')
        self.assertEqual(s.meta['last_rebalance'], '2026-06-03')
        self.assertAlmostEqual(s.nav_history.loc[s.nav_history.date == '2026-06-02','nav'].iloc[0],1.0201)
        self.assertFalse((self.root/'reports/rebalance/2026-06-01.md').exists())

    def test_official_index_is_the_benchmark(self):
        self.seed()
        self.invoke()
        self.assertAlmostEqual(State.load(self.root/'state').nav_history.iloc[-1]['benchmark_nav'],1.02)

    def test_corrupt_state_is_not_silently_reinitialized(self):
        s=self.seed()
        s.holdings.loc[0,'weight']=np.nan
        with self.assertRaises(ValueError): s.save()

    def test_missing_whole_historical_session_blocks_initialization(self):
        self.history=self.history[self.history.date!='2026-05-15']
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.root/'state/book.json').exists())

    def test_missing_turnover_is_not_averaged_away(self):
        self.history.loc[(self.history.date=='2026-06-02')&(self.history.symbol=='000001'),'turnover']=np.nan
        self.invoke()
        self.assertIn('000001',State.load(self.root/'state').meta['missing_symbols'])

    def test_holiday_does_not_revalue(self):
        self.seed()
        before=(self.root/'state/book.json').read_bytes()
        self.invoke('2026-06-06')
        self.assertEqual((self.root/'state/book.json').read_bytes(),before)

    def test_failing_monthly_solver_retains_drifted_holdings(self):
        self.seed(last='2026-05-29',rebalance='2026-05-04')
        cfg=json.loads((self.root/'config/strategy.json').read_text())
        cfg['optimizer']['max_weight']=.01  # 12 names cannot sum to one.
        (self.root/'config/strategy.json').write_text(json.dumps(cfg))
        self.invoke()
        s=State.load(self.root/'state')
        self.assertEqual(s.meta['last_rebalance'],'2026-05-04')
        self.assertAlmostEqual(s.nav_history.iloc[-1]['nav'],1.01**3)
        np.testing.assert_allclose(s.weights().values,1/12)

    def test_snapshot_uses_unadjusted_return_not_hfq_return(self):
        self.unadjusted_return=.02
        self.spot['pct_chg']=.02  # Valid raw market return differs from HFQ .01.
        self.invoke()
        self.assertTrue(State.load(self.root/'state').initialized)

    def test_genuinely_inconsistent_snapshot_still_blocks_rebalance(self):
        self.unadjusted_return=.02
        self.spot['pct_chg']=.03
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.root/'state/book.json').exists())

    def test_failed_current_constituent_is_reported_and_excluded(self):
        self.history=self.history[self.history.symbol!='000012']
        self.invoke()
        s=State.load(self.root/'state')
        self.assertIn('000012',s.meta['missing_symbols'])
        self.assertTrue((self.root/'reports/daily/2026-06-03.md').read_text().find('000012')>=0)


if __name__ == '__main__': unittest.main()
