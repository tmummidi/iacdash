"""Legacy UI regression checks. Run with requirements-dashboard.txt installed."""
import importlib.util
import sys
from pathlib import Path
import unittest
import contextlib
import io


AVAILABLE=all(importlib.util.find_spec(p) for p in ("dash","pandas","nltk","plotly"))


@unittest.skipUnless(AVAILABLE,"Install requirements-dashboard.txt to test the legacy dashboard")
class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/"src"/"app.py"
        spec=importlib.util.spec_from_file_location("legacy_iac_app",path)
        cls.app=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    def call(self,fn,*args):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args)

    def test_layout_serves_without_changing_working_directory(self):
        self.assertEqual(self.app.server.test_client().get("/_dash-layout").status_code,200)

    def test_empty_and_unknown_selection(self):
        for selection in (None,[],[999999]):
            self.assertEqual(len(self.call(self.app.update_output_for_sic,selection)),4)

    def test_empty_keyword(self):
        for keyword in (None,"","   ","["):
            self.assertEqual(self.call(self.app.update_output_for_keywords,keyword),[])

    def test_keyword_works_without_downloaded_nltk_corpora(self):
        import nltk
        prior=nltk.data.path
        try:
            nltk.data.path=[]
            self.assertGreater(len(self.call(self.app.update_output_for_keywords,"food")),0)
        finally:
            nltk.data.path=prior

    def test_existing_industry_callback(self):
        self.assertEqual(len(self.call(self.app.update_output_for_sic,[2011])),4)
