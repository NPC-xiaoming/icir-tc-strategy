"""Provider failure, coverage and adjustment compatibility regressions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import quotes

class QuoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.cache=Path(self.tmp.name)/'quotes.parquet'
        self.dates=pd.bdate_range('2026-10-01',periods=9)
        self.raw=pd.DataFrame({'date':self.dates,'symbol':'000001','close':np.arange(100.,109.),
                               'pct_chg':.01,'turnover':1.})
    def fetch(self, dates=None):
        return quotes.fetch_history(['000001'],start_date=self.dates[0],end_date=self.dates[-1],
            required_dates=self.dates if dates is None else dates,cache_path=self.cache,workers=1,pause=0,allow_partial=True)
    def seed(self):
        frame=self.raw.iloc[:6].copy(); frame['provider']='eastmoney'; frame['adjust']='hfq'
        frame.to_parquet(self.cache,index=False)
    def test_failed_primary_uses_compatible_secondary(self):
        self.seed()
        def provider(name,*args,**kw):
            if name=='eastmoney': raise ConnectionError('connection closed')
            return self.raw.copy()
        with patch.object(quotes,'_provider_frame',side_effect=provider),patch.object(quotes.time,'sleep'):
            out=self.fetch()
        self.assertEqual(len(out),9)
        self.assertEqual(out.attrs['failed_symbols'],[])
        self.assertEqual(out.attrs['sources']['000001'],'tencent')
        self.assertIn('connection closed',str(out.attrs['errors']))
    def test_incompatible_adjustment_is_not_used(self):
        self.seed()
        def provider(name,*args,**kw):
            if name=='eastmoney': raise ConnectionError('offline')
            frame=self.raw.copy(); frame['pct_chg']=.02; return frame
        with patch.object(quotes,'_provider_frame',side_effect=provider),patch.object(quotes.time,'sleep'):
            out=self.fetch()
        self.assertEqual(out.attrs['failed_symbols'],['000001'])
        self.assertEqual(len(out),0)  # incompatible cached backup is invalidated
    def test_all_providers_down_preserves_cache_and_errors(self):
        self.seed()
        with patch.object(quotes,'_provider_frame',side_effect=ConnectionError('offline')),patch.object(quotes.time,'sleep'):
            out=self.fetch()
        self.assertEqual(out.attrs['failed_symbols'],['000001'])
        self.assertEqual(len(pd.read_parquet(self.cache)),6)
        self.assertTrue(out.attrs['errors'])
    def test_cached_complete_window_does_not_hit_network(self):
        self.seed()
        with patch.object(quotes,'_provider_frame',side_effect=AssertionError('network called')) as fn:
            out=self.fetch(self.dates[:6])
        self.assertEqual(len(out),6); fn.assert_not_called()
    def test_stale_response_triggers_fallback(self):
        self.seed()
        def provider(name,*args,**kw):
            return self.raw.iloc[:6].copy() if name=='eastmoney' else self.raw.copy()
        with patch.object(quotes,'_provider_frame',side_effect=provider),patch.object(quotes.time,'sleep'):
            out=self.fetch()
        self.assertEqual(out.attrs['sources']['000001'],'tencent')
    def test_fallback_without_reference_is_rejected(self):
        def provider(name,*args,**kw):
            if name=='eastmoney': raise ConnectionError('offline')
            return self.raw.copy()
        with patch.object(quotes,'_provider_frame',side_effect=provider),patch.object(quotes.time,'sleep'):
            out=self.fetch()
        self.assertTrue(out.empty)
    def test_tencent_return_needs_immediately_previous_trading_day(self):
        raw=pd.DataFrame({'date':[self.dates[0],self.dates[2]],'close':[100.,121.]})
        with patch.object(quotes.ak,'stock_zh_a_hist_tx',return_value=raw):
            out=quotes._provider_frame('tencent','000001',self.dates[0],self.dates[2],'hfq',10,
                                       calendar=self.dates)
        self.assertNotIn(self.dates[2], out.date.tolist())
    def test_sina_turnover_converts_fraction_to_percentage(self):
        raw=pd.DataFrame({'date':self.dates[:2],'close':[100.,101.],'turnover':[.02,.03]})
        with patch.object(quotes.ak,'stock_zh_a_daily',return_value=raw):
            out=quotes._provider_frame('sina','000001',self.dates[0],self.dates[1],'hfq',10,calendar=self.dates)
        self.assertAlmostEqual(out.iloc[-1].pct_chg,.01)
        self.assertAlmostEqual(out.iloc[-1].turnover,3.)

    def test_retries_transient_failure_before_switching(self):
        with patch.object(quotes,'_provider_frame',side_effect=[ConnectionError('temporary'),self.raw.copy()]) as call,patch.object(quotes.time,'sleep'):
            out=self.fetch()
        self.assertEqual(call.call_count,2)
        self.assertEqual(out.attrs['sources']['000001'],'eastmoney')

    def test_primary_outage_opens_circuit(self):
        def provider(name,*args,**kw):
            raise ConnectionError('offline')
        with patch.object(quotes,'_provider_frame',side_effect=provider) as call,patch.object(quotes.time,'sleep'):
            out=quotes.fetch_history([f'{n:06d}' for n in range(20)],start_date=self.dates[0],end_date=self.dates[-1],
                required_dates=self.dates,workers=1,pause=0,allow_partial=True)
        self.assertLessEqual(call.call_count,18)
        self.assertEqual(len(out.attrs['failed_symbols']),20)

    def test_cached_backup_is_rechecked_after_primary_correction(self):
        ref=self.raw.iloc[:6].copy(); ref['provider']='eastmoney'; ref['adjust']='hfq'
        backup=self.raw.iloc[:8].copy(); backup['provider']='tencent'; backup['adjust']='hfq'
        pd.concat([ref,backup],ignore_index=True).to_parquet(self.cache,index=False)
        def provider(name,*args,**kw):
            frame=self.raw.copy(); frame['pct_chg']=.02 if name=='eastmoney' else .03
            return frame.iloc[:7] if name=='eastmoney' else frame
        with patch.object(quotes,'_provider_frame',side_effect=provider),patch.object(quotes.time,'sleep'):
            out=self.fetch()
        self.assertEqual(out.attrs['failed_symbols'],['000001'])
        with patch.object(quotes,'_provider_frame',side_effect=ConnectionError('offline')),patch.object(quotes.time,'sleep'):
            out=self.fetch(self.dates[:8])
        self.assertEqual(out.attrs['failed_symbols'],['000001'])
        self.assertNotIn('000001',out.attrs['sources'])

    def test_sina_primary_ignores_eastmoney_cache_and_needs_no_reference(self):
        self.seed()
        fresh=self.raw.copy(); fresh['pct_chg']=.02
        with patch.object(quotes,'_provider_frame',return_value=fresh) as fn,patch.object(quotes.time,'sleep'):
            out=quotes.fetch_history(['000001'],start_date=self.dates[0],end_date=self.dates[-1],
                required_dates=self.dates,cache_path=self.cache,workers=1,pause=0,
                primary_provider='sina')
        self.assertEqual(out.attrs['sources'],{'000001':'sina'})
        self.assertTrue((out.pct_chg==.02).all())
        self.assertEqual([c.args[0] for c in fn.call_args_list],['sina'])
        with patch.object(quotes,'_provider_frame',side_effect=AssertionError('network')) as fn:
            cached=quotes.fetch_history(['000001'],start_date=self.dates[0],end_date=self.dates[-1],
                required_dates=self.dates,cache_path=self.cache,primary_provider='sina')
        fn.assert_not_called()
        self.assertEqual(len(cached),9)

    def test_sina_failure_does_not_fall_back_to_other_source(self):
        self.seed()
        with patch.object(quotes,'_provider_frame',side_effect=ConnectionError('offline')) as fn,patch.object(quotes.time,'sleep'):
            out=quotes.fetch_history(['000001'],start_date=self.dates[0],end_date=self.dates[-1],
                required_dates=self.dates,cache_path=self.cache,primary_provider='sina',allow_partial=True)
        self.assertTrue(out.empty)
        self.assertTrue(all(c.args[0]=='sina' for c in fn.call_args_list))

    def test_sina_unadjusted_market_cap_uses_float_shares(self):
        raw=pd.DataFrame({'date':self.dates[:2],'close':[10.,11.],
            'turnover':[.02,.03],'outstanding_share':[1000.,1200.]})
        with patch.object(quotes.ak,'stock_zh_a_daily',return_value=raw):
            out=quotes._provider_frame('sina','000001',self.dates[0],self.dates[1],'',10,calendar=self.dates)
        self.assertEqual(out.iloc[-1].circ_mv,13200.)
