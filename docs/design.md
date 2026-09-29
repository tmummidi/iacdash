# Model and release design

## User and decision

Select an affordable, implementable portfolio of industrial energy measures. Intended users are analysts and operations researchers
evaluating this decision before committing operational resources.

## Formulation

~~~text
Each measure has capital cost, annual savings, life, labor and an alternative group.
Synthetic NPV scenario = discounted lifetime savings * correlated multiplier - cost.
Choose binary z(i), with capital and labor budgets and at most one alternative
per (facility, group). Maximize:

mean_s NPV_s(z) + risk_aversion * lower_CVaR_alpha(NPV(z)).
lower_CVaR = max_eta [eta - mean_s max(eta-NPV_s(z),0)/alpha].

Introduce one eta and scenario shortfall variables for a linear MILP.
The baseline greedily ranks positive mean NPV / capital cost while honoring the
same constraints. Evaluate both plans on independently generated savings scenarios.
~~~

## Data and provenance

Synthetic measures for benchmarks; legacy dataset kept separate with provenance limitations. The generator is the authoritative data source. Parameters
represent hypothetical examples, not estimates from any employer. Seeds,
configuration hashes and package versions are included with executed results.
No pre-existing code from a tutorial or employer is used in the new model.

## Architecture

The retained Dash application explores the legacy energy-audit CSVs. The new energyplan package uses SQLite constraints to validate measure input, NumPy to generate correlated savings scenarios, and SciPy/HiGHS for investment allocation. The two data paths are deliberately explicit.

Dependencies are intentionally small. NumPy handles arrays; SciPy provides
numerical/statistical routines and the free HiGHS solver where applicable.
Distributed infrastructure is not justified by these measurements.

## Evaluation and acceptance

Baseline: Savings-to-cost greedy selection.
Held-out correlated savings scenarios; NPV, lower-tail NPV, constraint checks.
Meaningful tests use analytical results, enumeration, input constraints or
conservation laws. Confidence intervals and primal feasibility are reported,
not inferred from successful process exit alone.

The first release is accepted when installation into a clean environment,
unit tests and an installed CLI example all succeed. The release-validation
log records that check separately from the benchmark timings.

## Scope and limits

The benchmark uses synthetic measures and one-time binary investment decisions with fixed life and discount rate. Savings share global and facility shocks. No cash-flow scheduling, endogenous energy prices, causal savings estimation or verified implementation impacts are modeled. Legacy datasets are not used for the new benchmark.

A no-budget case selects no investments. A solver time limit is labeled and any incumbent is checked; no incumbent falls back to the documented greedy policy. A CVaR objective reflects the supplied scenario distribution, not a guarantee against unmodeled shocks.

## Resource budget

CPU; controlled scenario count and time limit. The published benchmark is serial and uses three increasing
workloads. Very large configurations can exceed laptop memory or practical
solver time. Only the recorded configurations were executed.

## Extensions

Document the provenance and redistribution terms of every legacy CSV. Add verified measure acquisition and cash-flow schedules. Compare site diversification and risk weights across heldout scenarios. Extract the legacy dashboard callbacks into a modern app factory.

## Method references

- SciPy 1.13 MILP API and result status: https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.milp.html
- SimPy shared resources: https://simpy.readthedocs.io/en/stable/simpy_intro/shared_resources.html
- NIST reliability estimation with censoring: https://www.itl.nist.gov/div898/handbook/apr/section4/apr413.htm
- SciPy Gamma shape/scale parameterization: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

References describe underlying methods. Results in this repository come from
the included implementation and executable configurations.
