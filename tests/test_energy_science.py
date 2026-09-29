import itertools
import sqlite3
import unittest
import numpy as np
from energyplan.data import Measure,ingest,npv_scenarios
from energyplan.optimize import optimize,greedy,lower_cvar,validate_selection


class EnergyTests(unittest.TestCase):
    def measures(self):
        return [Measure(0,0,0,4,2,5,2),Measure(1,0,0,5,3,5,2),
                Measure(2,1,0,6,4,5,3),Measure(3,1,1,3,1,5,1)]

    def test_sql_constraints_reject_duplicates(self):
        with self.assertRaises(sqlite3.IntegrityError):ingest([self.measures()[0]]*2)
        with self.assertRaises(sqlite3.IntegrityError):ingest([Measure(0,0,0,-1,2,5,1)])

    def test_risk_objective_matches_exhaustive_enumeration(self):
        measures=self.measures()
        outcomes=np.array([[5,8,9,4],[4,2,-2,3],[6,9,12,5],[3,5,1,2]],float)
        selection,result=optimize(measures,outcomes,10,5,risk_aversion=.5,alpha=.25)
        scores=[]
        for x in itertools.product([0,1],repeat=4):
            if validate_selection(measures,x,10,5):
                v=outcomes@x;scores.append(v.mean()+.5*lower_cvar(v,.25))
        self.assertAlmostEqual(result["training_utility"],max(scores),places=6)
        self.assertTrue(validate_selection(measures,selection,10,5))

    def test_fractional_tail_mass(self):
        self.assertAlmostEqual(lower_cvar([1,2,3,4],.375),(1+.5*2)/1.5)

    def test_no_budget_means_no_investment(self):
        m=self.measures();s=npv_scenarios(m,1,8)
        x,_=optimize(m,s,0,10)
        self.assertEqual(x.sum(),0)
        self.assertEqual(greedy(m,s,0,10).sum(),0)

    def test_discount_zero_and_seed_reproducibility(self):
        m=self.measures()
        a=npv_scenarios(m,12,20,0);b=npv_scenarios(m,12,20,0)
        np.testing.assert_array_equal(a,b)

    def test_risk_neutral_at_least_greedy_training_value(self):
        m=self.measures();s=npv_scenarios(m,14,10)
        b=greedy(m,s,10,5);x,r=optimize(m,s,10,5,0)
        self.assertGreaterEqual(np.mean(s@x)+1e-7,np.mean(s@b))
