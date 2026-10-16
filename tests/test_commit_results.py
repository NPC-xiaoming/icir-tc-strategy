"""Exercise actual Git persistence against temporary local repositories."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/commit_results.sh'

class CommitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'work'; self.root.mkdir()
        self.remote=Path(self.tmp.name)/'remote.git'
        self.git('init','-b','main')
        self.git('config','user.name','test'); self.git('config','user.email','test@example.invalid')
        (self.root/'README.md').write_text('test')
        self.git('add','.'); self.git('commit','-m','initial')
        subprocess.run(['git','init','--bare',str(self.remote)],check=True,capture_output=True)
        self.git('remote','add','origin',str(self.remote)); self.git('push','-u','origin','main')

    def git(self,*args):
        return subprocess.run(['git',*args],cwd=self.root,check=True,capture_output=True,text=True).stdout.strip()

    def run_script(self):
        return subprocess.run(['bash',str(SCRIPT)],cwd=self.root,capture_output=True,text=True,
                              env={**os.environ,'DEFAULT_BRANCH':'main'})

    def test_absent_reports_is_successful_noop(self):
        before=self.git('rev-parse','HEAD')
        result=self.run_script()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.git('rev-parse','HEAD'),before)

    def test_state_and_reports_reach_remote(self):
        (self.root/'state').mkdir(); (self.root/'state/book.json').write_text('{}')
        (self.root/'reports').mkdir(); (self.root/'reports/day.md').write_text('report')
        result=self.run_script()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.git('show','origin/main:reports/day.md'),'report')
        self.assertEqual(self.git('status','--porcelain'),'')

    def test_failed_push_preserves_local_results(self):
        (self.root/'state').mkdir(); (self.root/'state/book.json').write_text('{}')
        self.git('remote','set-url','origin',str(self.remote/'missing'))
        result=self.run_script()
        self.assertNotEqual(result.returncode,0)
        self.assertEqual((self.root/'state/book.json').read_text(),'{}')
        self.assertEqual(self.git('show','HEAD:state/book.json'),'{}')

    def test_cache_only_commit_excludes_ledger_and_reports(self):
        (self.root/'state').mkdir(); (self.root/'reports').mkdir()
        (self.root/'state/market_history.parquet').write_bytes(b'cache')
        (self.root/'state/book.json').write_text('uncommitted valuation')
        (self.root/'reports/day.md').write_text('uncommitted report')
        result=subprocess.run(['bash',str(SCRIPT)],cwd=self.root,capture_output=True,text=True,
                              env={**os.environ,'DEFAULT_BRANCH':'main','CACHE_ONLY':'1'})
        self.assertEqual(result.returncode,0,result.stderr)
        tree=self.git('ls-tree','-r','--name-only','origin/main')
        self.assertIn('market_history.parquet',tree)
        self.assertNotIn('book.json',tree)
        self.assertNotIn('reports/',tree)
