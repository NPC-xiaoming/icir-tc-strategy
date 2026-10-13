"""Refresh the SW level-1 cache; --if-due performs a monthly after-close check."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import data
from src.state import atomic_text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--if-due',action='store_true')
    args=parser.parse_args()
    now=pd.Timestamp.now(tz='Asia/Shanghai')
    cache=ROOT/'state/industry_sw_l1.parquet'
    metadata=ROOT/'state/industry_meta.json'
    with data.network_timeout(20):
        if args.if_due:
            if now.hour<16:
                print('Industry refresh skipped before 16:00 Shanghai.'); return 0
            cal=data._retry(lambda:data.ak.tool_trade_date_hist_sina(),'trading calendar')
            dates=pd.to_datetime(cal.trade_date)
            today=now.tz_localize(None).normalize()
            if dates.max()<today: raise ValueError('Trading calendar does not cover today')
            if today not in set(dates):
                print('Industry refresh skipped on non-trading day.'); return 0
            if cache.exists() and metadata.exists():
                saved=json.loads(metadata.read_text())
                if saved.get('month')==now.strftime('%Y-%m'):
                    print('Industry cache already refreshed this month.'); return 0
        try:
            frame=data.refresh_industry(cache)
        except Exception as exc:
            if not cache.exists():
                raise
            print(f'WARNING: industry refresh failed; retaining existing cache: {exc}', file=sys.stderr)
            return 0
        atomic_text(metadata,json.dumps({'month':now.strftime('%Y-%m'),'refreshed_at':now.isoformat(),
            'symbols':len(frame),'industries':int(frame.industry.nunique())},ensure_ascii=False,indent=2))
    print(f'Saved {len(frame)} symbols / {frame.industry.nunique()} industries.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
