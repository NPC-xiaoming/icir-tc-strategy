"""Read-only provider probe using the same runtime as the daily workflow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import akshare as ak
import pandas as pd
from src.data import network_timeout

print(f'AKShare={ak.__version__}', flush=True)
end = pd.Timestamp.now(tz='Asia/Shanghai').normalize().tz_localize(None)
start = end - pd.Timedelta(days=10)
for symbol in ['000001', '600519']:
    market = ('sh' if symbol.startswith('6') else 'sz') + symbol
    for provider, fn in [
        ('eastmoney', lambda: ak.stock_zh_a_hist(symbol=symbol, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'), adjust='hfq', timeout=10)),
        ('tencent', lambda: ak.stock_zh_a_hist_tx(symbol=market, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'), adjust='hfq', timeout=10)),
        ('sina', lambda: ak.stock_zh_a_daily(symbol=market, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'), adjust='hfq')),
    ]:
        try:
            with network_timeout(10):
                frame = fn()
            print(f'{provider} {symbol}: rows={len(frame)}\n{frame.tail(5).to_string(index=False)}', flush=True)
        except Exception as exc:
            print(f'{provider} {symbol}: {type(exc).__name__}: {exc}', flush=True)
