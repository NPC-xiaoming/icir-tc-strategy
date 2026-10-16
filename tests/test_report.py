import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.report import make_nav_plot


class NavPlotTests(unittest.TestCase):
    def test_backtest_series_are_included_in_all_three_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = pd.DataFrame({
                'date': pd.to_datetime(['2026-10-12', '2026-10-13']),
                'nav': [1.0, 1.1],
                'benchmark_nav': [1.0, 1.02],
            })
            backtest = root / 'backtest.csv'
            pd.DataFrame({
                'date': pd.to_datetime(['2015-01-01', '2015-01-02', '2015-01-03']),
                'nav': [1.0, 1.1, 1.05],
                'benchmark_nav': [1.0, 1.02, 1.01],
            }).to_csv(backtest, index=False)

            output = root / 'chart.png'
            make_nav_plot(history, output, backtest)

            self.assertGreater(output.stat().st_size, 10_000)
            self.assertGreater(len(pd.read_csv(backtest).columns), 2)


if __name__ == '__main__':
    unittest.main()
