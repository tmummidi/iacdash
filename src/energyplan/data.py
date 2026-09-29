"""Validated measure ingestion. Legacy audit CSVs are not silently repurposed."""
import csv
from dataclasses import dataclass,asdict
import sqlite3
import numpy as np


@dataclass(frozen=True)
class Measure:
    id: int
    facility: int
    alternative_group: int
    cost: float
    annual_saving: float
    life_years: int
    labor_hours: float


def ingest(rows):
    """Validate with SQLite constraints and return deterministic row ordering."""
    db=sqlite3.connect(":memory:")
    try:
        db.execute("""CREATE TABLE measures(
            id INTEGER PRIMARY KEY, facility INTEGER NOT NULL, alternative_group INTEGER NOT NULL,
            cost REAL NOT NULL CHECK(cost>0), annual_saving REAL NOT NULL CHECK(annual_saving>=0),
            life_years INTEGER NOT NULL CHECK(life_years>0),
            labor_hours REAL NOT NULL CHECK(labor_hours>0))""")
        values=[]
        for r in rows:
            values.append(tuple(asdict(r).values()))
        if not values:
            raise ValueError("At least one measure required")
        if not np.isfinite(np.array(values,float)).all():
            raise ValueError("All measure fields must be finite")
        for v in values:
            if any(int(v[i])!=v[i] for i in [0,1,2,5]):
                raise ValueError("Identifiers and life_years must be integral")
        with db:
            db.executemany("INSERT INTO measures VALUES (?,?,?,?,?,?,?)",values)
        return [Measure(*r) for r in db.execute("SELECT * FROM measures ORDER BY id")]
    finally:
        db.close()


def read_csv(path):
    with open(path,newline="") as f:
        rows=[]
        for row in csv.DictReader(f):
            rows.append(Measure(**{k:int(v) if k in ("id","facility","alternative_group","life_years") else float(v) for k,v in row.items()}))
    return ingest(rows)


def synthetic(seed,n):
    if n<1:
        raise ValueError("Positive measure count required")
    rng=np.random.default_rng(seed)
    return ingest([Measure(i,i//10,(i%10)//2,float(rng.integers(5000,40000)),
        float(rng.uniform(2000,16000)),int(rng.integers(5,13)),float(rng.integers(8,100))) for i in range(n)])


def npv_scenarios(measures,seed,count,discount=.07):
    if count<1 or discount<0:
        raise ValueError("Invalid scenario count or discount")
    rng=np.random.default_rng(seed)
    facilities={f:i for i,f in enumerate(sorted({m.facility for m in measures}))}
    common=rng.lognormal(-.5*.12**2,.12,(count,1))
    site=rng.lognormal(-.5*.25**2,.25,(count,len(facilities)))
    individual=rng.lognormal(-.5*.10**2,.10,(count,len(measures)))
    factors=np.array([m.life_years if discount==0 else (1-(1+discount)**(-m.life_years))/discount for m in measures])
    savings=np.array([m.annual_saving for m in measures])*factors
    multipliers=common*site[:,[facilities[m.facility] for m in measures]]*individual
    return multipliers*savings-np.array([m.cost for m in measures])
