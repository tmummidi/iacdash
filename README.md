# IAC Decision Lab: energy audits and investment selection

**Decision:** Select an affordable, implementable portfolio of industrial energy measures.

Measures compete for money and implementation labor; alternatives at one facility cannot all be installed. Correlated savings make selecting only the largest mean ROI fragile.

This is a working v0.1 research engineering release, with synthetic data, a free
solver path, scientific tests, and recorded local measurements. It is independent
portfolio work. Employer systems and impact numbers are not benchmark evidence.

## Run

Python 3.10 or 3.12 is the supported CI target; the recorded local verification
used Python 3.12.14.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m energyplan --output results/demo.json
python -m unittest discover -s tests -v
python -m energyplan --benchmark --output results/benchmark.json
~~~

On Windows activate with .venv\Scripts\activate. The core has no paid service,
proprietary solver license, external dataset download, or GPU requirement.
Package dependencies and build tools are pinned in pyproject.toml.

To use a configuration:

~~~bash
python -m energyplan --config configs/ci.json --output results/small.json
~~~

Run energyplan with synthetic data or pass csv_path in JSON. Required CSV columns: id, facility, alternative_group, cost, annual_saving, life_years, labor_hours. Duplicate IDs, invalid dimensions and invalid numeric values are rejected. Run the legacy dashboard separately with requirements-dashboard.txt.

## Implementation

The retained Dash application explores the legacy energy-audit CSVs. The new energyplan package uses SQLite constraints to validate measure input, NumPy to generate correlated savings scenarios, and SciPy/HiGHS for investment allocation. The two data paths are deliberately explicit.

Baseline: Savings-to-cost greedy selection. Evaluation: Held-out correlated savings scenarios; NPV, lower-tail NPV, constraint checks.
The full mathematical specification is in [docs/design.md](docs/design.md).

## Executed measurements

| Workload | Runtime (s) | Peak RSS (MiB) | Measured quality |
|---|---:|---:|---|
| 25 measures / 32 training scenarios | 0.010 | 88.9 | mean synthetic NPV +5.6%; gap 0.0% |
| 100 measures / 32 training scenarios | 0.019 | 91.1 | mean synthetic NPV +13.4%; gap 0.0% |
| 400 measures / 32 training scenarios | 0.082 | 103.2 | mean synthetic NPV +12.2%; gap 0.0% |

The 400-measure case raised mean synthetic NPV by 772,313 monetary units, with a paired 95% interval [761,086, 783,541]. Lower-tail NPV rose from 4,560,871 to 5,073,527. These are generated scenarios, not measured industrial savings.

Measurements ran on an AMD EPYC 9V74 host in a Linux container with an 8-vCPU,
8-GiB cgroup limit. These are laptop-sized workloads, not laptop hardware
measurements. One fresh process was used per scale case. Runtime includes
generation, fitting/optimization and evaluation, but excludes Python imports;
the report also records cold-process time. Peak RSS includes imports.
BLAS/OpenMP thread environment defaults were set to one; actual solver thread
utilization was not measured. No distributed scaling or parallel speedup is claimed.
Single timing observations are not latency distributions.

Raw seeds, versions, configurations, peak RSS, solver/statistical evidence and
derived throughput are in [results/benchmark.json](results/benchmark.json).
The throughput numerator is project-specific and includes only the declared work
units; it must not be compared across projects.

## Assumptions and failure cases

The benchmark uses synthetic measures and one-time binary investment decisions with fixed life and discount rate. Savings share global and facility shocks. No cash-flow scheduling, endogenous energy prices, causal savings estimation or verified implementation impacts are modeled. Legacy datasets are not used for the new benchmark.

A no-budget case selects no investments. A solver time limit is labeled and any incumbent is checked; no incumbent falls back to the documented greedy policy. A CVaR objective reflects the supplied scenario distribution, not a guarantee against unmodeled shocks.

## Reproducibility and contribution

Data are generated from explicit seeds. Training and heldout scenarios are
separate; confidence intervals describe evaluation sampling variability, not
uncertainty in all model assumptions. Larger configurations in
configs/larger-unexecuted.json are **unexecuted** and are not performance claims.
No hypotheses are presented as measurements.

See [CONTRIBUTING.md](CONTRIBUTING.md), [THIRD_PARTY.md](THIRD_PARTY.md), and
[LICENSE](LICENSE). v0.1 covers the complete single-machine vertical slice above.
It is not a production deployment.

## Retained dashboard

The pre-existing energy-audit dashboard and its CSV files remain in src/.
Repairs make CSV paths independent of the working directory, handle empty
selections, and remove downloadable NLTK corpus requirements from keyword search.
The keyword tokenizer uses an explicit small stop-word set and Porter stemming.

~~~bash
python -m pip install -r requirements-dashboard.txt
python src/app.py
~~~

The dashboard depends on legacy CSV redistribution/provenance that is still
unresolved. Its data are not newly relicensed and are not consumed by the
synthetic energyplan benchmark. Existing history and notebook work are preserved.
The MIT grant covers the new package, tests and documentation only.
