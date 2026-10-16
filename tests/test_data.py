import unittest
from unittest.mock import patch

import pandas as pd
from src import data


class DataTests(unittest.TestCase):
    def test_partial_history_download_fails(self):
        raw=pd.DataFrame({'日期':['2026-06-03'],'收盘':[10.], '涨跌幅':[1.],'换手率':[1.]})
        with patch.object(data.ak,'stock_zh_a_hist',side_effect=lambda **kw: raw if kw['symbol']=='000001' else pd.DataFrame()), \
             patch.object(data.ak,'stock_zh_a_hist_tx',return_value=pd.DataFrame()), \
             patch.object(data.ak,'stock_zh_a_daily',return_value=pd.DataFrame()), \
             patch.object(data.time,'sleep'):
            with self.assertRaises(RuntimeError):
                data.fetch_stock_history(['000001','000002'],pause=0,start_date='2026-06-03',end_date='2026-06-03')

    def test_bad_weight_data_is_not_renormalized(self):
        raw=pd.DataFrame({'成分券代码':['000001','000002'],'成分券名称':['A','B'],
                          '权重':[50.,float('nan')],'日期':['2026-05-29']*2})
        with patch.object(data.ak,'index_stock_cons_weight_csindex',return_value=raw):
            with self.assertRaises(ValueError): data.fetch_constituents()

    def test_partial_history_can_be_recorded_for_retry(self):
        raw=pd.DataFrame({'日期':['2026-06-03'],'收盘':[10.], '涨跌幅':[1.],'换手率':[1.]})
        with patch.object(data.ak,'stock_zh_a_hist',side_effect=lambda **kw: raw if kw['symbol']=='000001' else pd.DataFrame()), \
             patch.object(data.ak,'stock_zh_a_hist_tx',return_value=pd.DataFrame()), \
             patch.object(data.ak,'stock_zh_a_daily',return_value=pd.DataFrame()), \
             patch.object(data.time,'sleep'):
            frame=data.fetch_stock_history(['000001','000002'],pause=0,allow_partial=True,start_date='2026-06-03',end_date='2026-06-03')
        self.assertEqual(frame.attrs['failed_symbols'],['000002'])
        self.assertEqual(frame.symbol.tolist(),['000001'])

    def test_stale_index_response_tries_another_provider(self):
        required=pd.to_datetime(['2026-10-14','2026-10-15'])
        stale=pd.DataFrame({'date':required[:1],'close':[100.]})
        fresh=pd.DataFrame({'date':required,'close':[100.,101.]})
        with patch.object(data.ak,'stock_zh_index_daily_em',return_value=stale), \
             patch.object(data.ak,'stock_zh_index_daily_tx',return_value=fresh), \
             patch.object(data.time,'sleep'):
            got=data.fetch_index_history(start_date=required[0],end_date=required[-1],required_dates=required)
        self.assertEqual(got.attrs['provider'],'tencent')
        self.assertEqual(len(got),2)



    def test_fixed_sina_index_does_not_use_other_providers(self):
        required=pd.to_datetime(['2026-10-14','2026-10-15'])
        frame=pd.DataFrame({'date':required,'close':[100.,101.]})
        with patch.object(data.ak,'stock_zh_index_daily',return_value=frame), \
             patch.object(data.ak,'stock_zh_index_daily_em') as em, \
             patch.object(data.ak,'stock_zh_index_daily_tx') as tx:
            got=data.fetch_index_history(start_date=required[0],end_date=required[-1],
                                         required_dates=required,provider='sina')
        em.assert_not_called();tx.assert_not_called()
        self.assertEqual(got.attrs['provider'],'sina')

    def test_sina_snapshot_uses_dated_raw_cap(self):
        raw=pd.DataFrame({'date':pd.to_datetime(['2026-10-16']),'symbol':['000001'],
            'provider':['sina'],'adjust':[''],'close':[11.5],'pct_chg':[.01],
            'turnover':[.5],'circ_mv':[123456.]})
        with patch.object(data.ak,'stock_zh_a_spot_em') as em:
            got=data.fetch_spot(history=raw)
        em.assert_not_called()
        self.assertEqual(got.iloc[0].circ_mv,123456.)
        raw['adjust']='hfq'
        with self.assertRaises(ValueError): data.fetch_spot(history=raw)
