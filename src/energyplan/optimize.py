"""Binary budgeted investment with mutually exclusive alternatives and CVaR."""
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix


def lower_cvar(values,alpha=.1):
    v=np.sort(np.asarray(values,float))
    if v.ndim!=1 or len(v)==0 or not np.isfinite(v).all() or not 0<alpha<=1:
        raise ValueError("Finite outcomes and tail fraction in (0,1] required")
    mass=alpha*len(v);whole=int(np.floor(mass));fraction=mass-whole
    return float((v[:whole].sum()+(fraction*v[whole] if whole<len(v) else 0))/mass)


def validate_selection(measures,selection,budget,hours):
    x=np.asarray(selection,float)
    if x.shape!=(len(measures),) or not np.isin(x,[0,1]).all():
        return False
    picked=[m for m,z in zip(measures,x) if z]
    groups=[(m.facility,m.alternative_group) for m in picked]
    return (sum(m.cost for m in picked)<=budget+1e-6 and
            sum(m.labor_hours for m in picked)<=hours+1e-6 and len(set(groups))==len(groups))


def greedy(measures,npv,budget,hours):
    utility=np.mean(npv,axis=0)
    order=np.argsort(-utility/np.array([m.cost for m in measures]),kind="stable")
    x=np.zeros(len(measures),int)
    for i in order:
        if utility[i]<=0:
            continue
        trial=x.copy();trial[i]=1
        if validate_selection(measures,trial,budget,hours):
            x=trial
    return x


def optimize(measures,npv,budget,hours,risk_aversion=.5,alpha=.1,time_limit=10):
    npv=np.asarray(npv,float)
    if npv.ndim!=2 or npv.shape[1]!=len(measures) or len(npv)==0 or not np.isfinite(npv).all():
        raise ValueError("Finite scenario-by-measure NPV required")
    if min(budget,hours,risk_aversion)<0 or not 0<alpha<=1:
        raise ValueError("Invalid budget or risk settings")
    n=len(measures);s=len(npv);risk=risk_aversion>0
    groups=sorted({(m.facility,m.alternative_group) for m in measures})
    variables=n+(s+1 if risk else 0)
    rows=2+len(groups)+(s if risk else 0)
    a=lil_matrix((rows,variables))
    lower=np.full(rows,-np.inf);upper=np.zeros(rows)
    a[0,:n]=[m.cost for m in measures];upper[0]=budget
    a[1,:n]=[m.labor_hours for m in measures];upper[1]=hours
    for j,g in enumerate(groups):
        a[2+j,:n]=[int((m.facility,m.alternative_group)==g) for m in measures]
        upper[2+j]=1
    c=np.zeros(variables);c[:n]=-npv.mean(axis=0)
    lb=np.zeros(variables);ub=np.ones(variables)
    integrality=np.zeros(variables);integrality[:n]=1
    if risk:
        eta=n;z=np.arange(n+1,n+1+s)
        lb[eta]=-np.inf;ub[n:]=np.inf
        c[eta]=-risk_aversion;c[z]=risk_aversion/(alpha*s)
        for scenario in range(s):
            row=2+len(groups)+scenario
            a[row,:n]=-npv[scenario]
            a[row,eta]=1;a[row,z[scenario]]=-1
    result=milp(c,integrality=integrality,bounds=Bounds(lb,ub),
                constraints=LinearConstraint(a.tocsr(),lower,upper),
                options={"time_limit":float(time_limit),"mip_rel_gap":1e-6})
    if result.x is None:
        selection=greedy(measures,npv,budget,hours)
        status="fallback_greedy_no_incumbent";gap=None
    else:
        if np.max(np.abs(result.x[:n]-np.round(result.x[:n])))>1e-5:
            raise RuntimeError("Nonintegral solver incumbent")
        selection=np.round(result.x[:n]).astype(int)
        status="optimal" if result.status==0 else "feasible_time_limit"
        gap=float(result.mip_gap)
    if not validate_selection(measures,selection,budget,hours):
        raise RuntimeError("Returned investment plan violates operational constraints")
    outcomes=npv@selection
    return selection,{
        "status":status,"solver_status":int(result.status),"solver_message":result.message,
        "mip_gap":gap,"variables":variables,"constraints":rows,
        "selected_ids":[m.id for m,z in zip(measures,selection) if z],
        "cost":float(np.array([m.cost for m in measures])@selection),
        "labor_hours":float(np.array([m.labor_hours for m in measures])@selection),
        "training_mean_npv":float(outcomes.mean()),
        "training_lower_cvar":lower_cvar(outcomes,alpha),
        "training_utility":float(outcomes.mean()+risk_aversion*lower_cvar(outcomes,alpha))}
