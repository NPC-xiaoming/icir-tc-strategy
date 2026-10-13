import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimizer import optimize


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'config/strategy.json').read_text())
        self.ids=[str(i) for i in range(12)]
        self.bench=pd.Series(1/12,index=self.ids)
        self.old=pd.Series([.095]*10+[.025,.025],index=self.ids)
        self.cov=pd.DataFrame(np.eye(12)*.0001,index=self.ids,columns=self.ids)
        self.score=pd.Series(np.arange(12),index=self.ids,dtype=float)
        self.size=pd.Series(np.linspace(-1,1,12),index=self.ids)

    def test_infeasible_problem_keeps_old_holdings(self):
        self.cfg['optimizer']['turnover_theta']=0.
        self.cfg['optimizer']['deviation']=0.
        result=optimize(self.score,self.bench,self.cov,self.cfg,pretrade=self.old,size_z=self.size)
        self.assertTrue(result['fallback'])
        np.testing.assert_allclose(result['weights'].reindex(self.ids),self.old)
        self.assertEqual(result['cost'],0.)

    def test_enabled_size_constraint_cannot_be_silently_omitted(self):
        with self.assertRaises(ValueError):
            optimize(self.score,self.bench,self.cov,self.cfg,pretrade=self.old,size_z=None)

    def test_size_exposure_is_constrained(self):
        result=optimize(self.score,self.bench,self.cov,self.cfg,size_z=self.size)
        self.assertFalse(result['fallback'])
        self.assertLessEqual(abs(float((result['weights']-self.bench)@self.size)),.02001)

    def test_removed_names_count_against_turnover_budget(self):
        old=self.bench*.8
        old['removed']=.2
        result=optimize(self.score*0,self.bench,self.cov,self.cfg,pretrade=old,size_z=self.size)
        self.assertTrue(result['fallback'])  # sell .2 + buy .2 exceeds .25
        pd.testing.assert_series_equal(result['weights'],old)

    def test_missing_covariance_is_rejected(self):
        self.cov.iloc[0,0]=np.nan
        with self.assertRaises(ValueError):
            optimize(self.score,self.bench,self.cov,self.cfg,size_z=self.size)

    def test_conflicting_single_name_bounds_do_not_get_relaxed(self):
        bench=self.bench.copy(); bench.iloc[0]=.5; bench.iloc[1:]=.5/11
        result=optimize(self.score,bench,self.cov,self.cfg,size_z=self.size)
        self.assertTrue(result['fallback'])


if __name__ == '__main__': unittest.main()
