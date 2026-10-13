import unittest
from unittest.mock import patch

import pandas as pd
from src import data


class DataTests(unittest.TestCase):
    def test_partial_history_download_fails(self):
        raw=pd.DataFrame({'日期':['2026-06-03'],'收盘':[10.], '涨跌幅':[1.],'换手率':[1.]})
        with patch.object(data.ak,'stock_zh_a_hist',side_effect=[raw,pd.DataFrame()]), \
             patch.object(data.time,'sleep'):
            with self.assertRaises(RuntimeError):
                data.fetch_stock_history(['000001','000002'],pause=0)

    def test_bad_weight_data_is_not_renormalized(self):
        raw=pd.DataFrame({'成分券代码':['000001','000002'],'成分券名称':['A','B'],
                          '权重':[50.,float('nan')],'日期':['2026-05-29']*2})
        with patch.object(data.ak,'index_stock_cons_weight_csindex',return_value=raw):
            with self.assertRaises(ValueError): data.fetch_constituents()


if __name__ == '__main__': unittest.main()
