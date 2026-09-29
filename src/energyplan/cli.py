from dataclasses import asdict
import numpy as np
from .data import synthetic,npv_scenarios,read_csv
from .optimize import optimize,greedy,lower_cvar
from .evidence import cli,paired_interval

DEFAULT=dict(n=50,train_scenarios=32,test_scenarios=1000,budget_fraction=.2,hours_fraction=.25,
             risk_aversion=.5,alpha=.1,discount=.07,time_limit=10,seed=20260929,csv_path=None)
SCALES=[dict(n=n,train_scenarios=32,time_limit=5) for n in (25,100,400)]


def run(c):
    """Ingest measures, choose investments, and evaluate uncertain savings."""
    if c["test_scenarios"]<2 or not 0<=c["budget_fraction"]<=1 or not 0<=c["hours_fraction"]<=1:
        raise ValueError("Invalid evaluation count or resource fraction")
    seeds=[int(x) for x in np.random.SeedSequence(c["seed"]).generate_state(3)]
    measures=read_csv(c["csv_path"]) if c["csv_path"] else synthetic(seeds[0],c["n"])
    training=npv_scenarios(measures,seeds[1],c["train_scenarios"],c["discount"])
    budget=sum(m.cost for m in measures)*c["budget_fraction"]
    hours=sum(m.labor_hours for m in measures)*c["hours_fraction"]
    baseline=greedy(measures,training,budget,hours)
    selected,solver=optimize(measures,training,budget,hours,c["risk_aversion"],c["alpha"],c["time_limit"])
    test=npv_scenarios(measures,seeds[2],c["test_scenarios"],c["discount"])
    b=test@baseline;o=test@selected
    return {"work_units":{"count":int(test.size),"unit":"heldout measure-scenario pairs"},"project":"energyplan","data":"user CSV" if c["csv_path"] else "synthetic industrial measures; no empirical savings claims",
            "seeds":{"data":seeds[0],"training":seeds[1],"evaluation":seeds[2]},
            "measures":[asdict(m) for m in measures],"budget":budget,"labor_budget":hours,
            "solver":solver,"baseline_selected_ids":[m.id for m,z in zip(measures,baseline) if z],
            "heldout":{"baseline_mean_npv":float(b.mean()),"optimized_mean_npv":float(o.mean()),
                "baseline_lower_cvar":lower_cvar(b,c["alpha"]),"optimized_lower_cvar":lower_cvar(o,c["alpha"]),
                "optimized_minus_baseline_npv":paired_interval(o-b),
                "optimized_probability_negative_npv":float(np.mean(o<0))},
            "evaluated_measure_scenarios":int(test.size)}


def main():
    cli(run,DEFAULT,SCALES,"energyplan")
