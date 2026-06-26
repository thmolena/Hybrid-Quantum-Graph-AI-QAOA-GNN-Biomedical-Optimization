#!/usr/bin/env python3
"""
Rigorous OOD + geometric-graph experiment for UQ-QAOA (v3).

Addresses every red-team finding:
  (1) MATCHED-ANCHOR headline: primary delta = uq_qaoa - anchors_fixed, which
      isolates ONLY the uncertainty-scaled step geometry (identical 5-anchor
      budget, identical refinement). Also report uq_qaoa - tqa_refine (pure
      physics). The oracle max(tqa,knn) baseline is reported only as context.
  (2) GENUINE-OOD verification: compute each test instance's nearest-training
      normalized feature distance; threshold = 95th pct of in-distribution
      held-out distances. Report fraction genuinely OOD per family; OOD claims
      restricted to genuinely-OOD instances.
  (3) SATURATION handling: per-instance optimal-cut fraction (c_max/|E|) and
      achievable headroom; perfect-cut (bipartite) and ceiling (wheel) families
      flagged and reported separately, excluded from the headline pool.
  (4) Paired stats: win-rate, paired SIGN test (binomial), Cohen's d_z, plus a
      family-STRATIFIED hierarchical bootstrap; Holm-Bonferroni across the
      pre-registered family of tests. Median delta + IQR reported.
  (5) >=30 instances per family.
  (6) GEOMETRIC PROBE: grid/triangular/hex/rgg lattices, where TQA's linear
      ramp may be suboptimal and a learned center might genuinely help.

All evaluation uses the real exact-statevector evaluator and the real trained
GIN / 4-source posterior from uq_qaoa_core. Outcome reported honestly.
"""
from __future__ import annotations
import os, sys, json, math, pathlib
import numpy as np

CORE_DIR = pathlib.Path(
    "/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/"
    "Hybrid-Quantum-Graph-AI-QAOA-GNN-Biomedical-Optimization/submission/code")
sys.path.insert(0, str(CORE_DIR))
import uq_qaoa_core as core
from uq_qaoa_core import (DEPTH, graph_features, qaoa_cost_values, qaoa_ratio,
                          predict_gaussian, build_training_library, stable_seed)

DIM = 2 * DEPTH

# ----------------------------------------------------------------------
# Graph generators
# ----------------------------------------------------------------------
def _edge(u, v): return (min(u, v), max(u, v))

def grid_edges(n, seed):
    r = int(round(math.sqrt(n))); c = int(math.ceil(n / r)); es = set()
    for i in range(r):
        for j in range(c):
            idx = i*c+j
            if idx >= n: continue
            if j+1 < c and i*c+(j+1) < n: es.add(_edge(idx, i*c+(j+1)))
            if i+1 < r and (i+1)*c+j < n: es.add(_edge(idx, (i+1)*c+j))
    return sorted(es)

def triangular_edges(n, seed):
    r = int(round(math.sqrt(n))); c = int(math.ceil(n / r)); es = set()
    for i in range(r):
        for j in range(c):
            idx = i*c+j
            if idx >= n: continue
            if j+1 < c and i*c+(j+1) < n: es.add(_edge(idx, i*c+(j+1)))
            if i+1 < r and (i+1)*c+j < n: es.add(_edge(idx, (i+1)*c+j))
            # diagonal -> triangles
            if i+1 < r and j+1 < c and (i+1)*c+(j+1) < n: es.add(_edge(idx,(i+1)*c+(j+1)))
    return sorted(es)

def hex_edges(n, seed):
    # brick-wall hexagonal-ish lattice (degree ~3)
    r = int(round(math.sqrt(n))); c = int(math.ceil(n / r)); es = set()
    for i in range(r):
        for j in range(c):
            idx = i*c+j
            if idx >= n: continue
            if j+1 < c and i*c+(j+1) < n: es.add(_edge(idx, i*c+(j+1)))
            if i+1 < r and (i+1)*c+j < n and ((i+j) % 2 == 0): es.add(_edge(idx,(i+1)*c+j))
    # ensure no isolated
    deg = {}
    for u,v in es: deg[u]=deg.get(u,0)+1; deg[v]=deg.get(v,0)+1
    for v in range(n):
        if deg.get(v,0)==0 and v>0: es.add(_edge(v, v-1))
    return sorted(es)

def rgg_edges(n, seed):
    rng = np.random.RandomState(seed); pts = rng.rand(n, 2)
    radius = math.sqrt(2.2/(math.pi*n))*1.6; es=set()
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i]-pts[j]) <= radius: es.add((i,j))
    if len(es) < n-1:
        for i in range(n):
            d = np.linalg.norm(pts-pts[i], axis=1); d[i]=np.inf
            es.add(_edge(i, int(np.argmin(d))))
    return sorted(es)

def star_edges(n, seed):
    return sorted(_edge(0, i) for i in range(1, n))

def barbell_edges(n, seed):
    a = n // 2
    es = set()
    for i in range(a):
        for j in range(i+1, a): es.add((i, j))
    for i in range(a, n):
        for j in range(i+1, n): es.add((i, j))
    es.add(_edge(a-1, a))  # bridge
    return sorted(es)

def dense_edges(n, seed):
    rng = np.random.RandomState(seed); es=set()
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.85: es.add((i, j))
    return sorted(es)

def powerlaw_cluster_edges(n, seed):
    rng = np.random.RandomState(seed); m=2; p_tri=0.5
    adj=[set() for _ in range(n)]
    for i in range(m):
        for j in range(i+1,m): adj[i].add(j); adj[j].add(i)
    deg=np.zeros(n)
    for i in range(m): deg[i]=m-1
    for new in range(m,n):
        targets=set(); existing=np.arange(new)
        s=deg[:new].sum(); pr=deg[:new]/s if s>0 else np.ones(new)/new
        first=int(rng.choice(existing,p=pr)); targets.add(first)
        while len(targets)<m:
            if rng.random()<p_tri and adj[first]:
                cand=[x for x in adj[first] if x not in targets and x<new]
                if cand: targets.add(int(rng.choice(cand))); continue
            targets.add(int(rng.choice(existing,p=pr)))
        for t in targets: adj[new].add(t); adj[t].add(new); deg[new]+=1; deg[t]+=1
    es=set()
    for i in range(n):
        for j in adj[i]: es.add(_edge(i,j))
    return sorted(es)

def bipartite_edges(n, seed):
    rng=np.random.RandomState(seed); a=n//2
    left=list(range(a)); right=list(range(a,n)); es=set()
    for u in left:
        for v in right:
            if rng.random()<0.45: es.add((u,v))
    for v in range(n):
        if not any(v in e for e in es):
            es.add(_edge(v, rng.choice(right)) if v<a else _edge(rng.choice(left), v))
    return sorted(es)

def wheel_edges(n, seed):
    es=set(); hub=n-1
    for i in range(n-1):
        es.add(_edge(i,(i+1)%(n-1))); es.add(_edge(i,hub))
    return sorted(es)

# Family registry with metadata: kind in {geometric, structural, trivial}
OOD_FAMILIES = {
    "GRID":       (grid_edges,       "geometric"),
    "TRIANGULAR": (triangular_edges, "geometric"),
    "HEX":        (hex_edges,        "geometric"),
    "RGG":        (rgg_edges,        "geometric"),
    "STAR":       (star_edges,       "structural"),
    "BARBELL":    (barbell_edges,    "structural"),
    "DENSE":      (dense_edges,      "structural"),
    "POWERLAW":   (powerlaw_cluster_edges, "structural"),
    "BIPARTITE":  (bipartite_edges,  "trivial"),   # perfect cut exists
    "WHEEL":      (wheel_edges,      "trivial"),    # QAOA-saturated
}
IND_FAMILIES = ["ER", "REG", "BA", "WS"]

# ----------------------------------------------------------------------
# Refinement (identical for all methods)
# ----------------------------------------------------------------------
def _clip(theta, depth=DEPTH):
    t=theta.copy(); t[:depth]=np.clip(t[:depth],0,np.pi); t[depth:]=np.clip(t[depth:],0,np.pi); return t

def refine_from_anchors(anchors, deltas, edges, n, c_vals, budget, depth=DEPTH):
    ratios=[]; cands=[]
    def _eval(theta):
        r=qaoa_ratio(theta[:depth],theta[depth:],edges,n,c_vals); cands.append(theta); ratios.append(r); return r
    for a in anchors:
        if len(cands)>=budget: break
        if any(np.allclose(a,c,atol=1e-12) for c in cands): continue
        _eval(a)
    if not ratios: _eval(anchors[0])
    bi=int(np.argmax(ratios)); best_theta=cands[bi].copy(); best_ratio=ratios[bi]
    delta=deltas.copy(); remaining=budget-len(cands)
    while remaining>0:
        improved=False
        for j in range(2*depth):
            if remaining<=0: break
            for sign in (+1.0,-1.0):
                if remaining<=0: break
                probe=_clip(best_theta.copy()); probe[j]+=sign*delta[j]; probe=_clip(probe)
                r=_eval(probe); remaining-=1
                if r>best_ratio: best_ratio=r; best_theta=probe.copy(); improved=True; break
        if not improved: delta=delta*0.5
    return best_ratio

def _tqa(depth=DEPTH):
    T=0.75*depth; dt=T/depth; s=(np.arange(depth)+0.5)/depth
    return np.concatenate([dt*s, dt*(1-s)])

def methods_for(n, edges, features, depth=DEPTH):
    post_anchors, sigma2_post = core._build_uq_qaoa_posterior(n, edges, features, depth)
    theta_tqa, mu_post, mu_gin, mu_local, mu_global = post_anchors
    FIXED=np.full(DIM,0.15); delta_uq=np.clip(0.5*np.sqrt(sigma2_post),0.05,0.30)
    all5=[mu_post, theta_tqa, mu_gin, mu_local, mu_global]
    return {
        "tqa_refine":   ([theta_tqa], FIXED),
        "knn_refine":   ([mu_local], FIXED),
        "post_fixed":   ([mu_post], FIXED),       # learned center, fixed step
        "anchors_fixed":(all5, FIXED),            # matched anchors, fixed step (control)
        "uq_qaoa":      (all5, delta_uq),         # full method
    }
METHODS=["tqa_refine","knn_refine","post_fixed","anchors_fixed","uq_qaoa"]

# ----------------------------------------------------------------------
# Feature-distance (genuine OOD) utilities
# ----------------------------------------------------------------------
def build_feature_reference(depth=DEPTH):
    lib=build_training_library(depth=depth)
    feats=np.array([e["features"] for e in lib])
    std=feats.std(axis=0); std[std<1e-8]=1.0
    return feats, std

def nearest_train_dist(features, lib_feats, std):
    d=np.linalg.norm((lib_feats-features)/std, axis=1)
    return float(d.min())

# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------
def binom_two_sided_p(k, nn):
    # exact two-sided sign test p-value for k successes of nn (p=0.5)
    from math import comb
    if nn==0: return 1.0
    def tail(x): return sum(comb(nn,i) for i in range(0,x+1))/2**nn
    kk=min(k, nn-k)
    return min(1.0, 2*tail(kk))

def paired_stats(d):
    d=np.asarray(d,float); nn=len(d)
    wins=int(np.sum(d>1e-9)); losses=int(np.sum(d<-1e-9)); ties=nn-wins-losses
    dz=float(d.mean()/d.std()) if d.std()>0 else 0.0
    p=binom_two_sided_p(wins, wins+losses)
    return {"mean":float(d.mean()),"median":float(np.median(d)),
            "iqr":[float(np.percentile(d,25)),float(np.percentile(d,75))],
            "win_rate":wins/nn,"wins":wins,"losses":losses,"ties":ties,
            "cohens_dz":dz,"sign_p":p,"n":nn}

def strat_bootstrap_ci(deltas_by_family, nboot=10000, seed=11):
    rng=np.random.RandomState(seed); fams=list(deltas_by_family.keys())
    means=[]
    for _ in range(nboot):
        rf=rng.choice(len(fams), size=len(fams), replace=True)
        vals=[]
        for fi in rf:
            arr=deltas_by_family[fams[fi]]
            idx=rng.choice(len(arr), size=len(arr), replace=True)
            vals.append(np.mean(np.asarray(arr)[idx]))
        means.append(np.mean(vals))
    return float(np.percentile(means,2.5)), float(np.percentile(means,97.5))

def holm(pairs):
    # pairs: list of (label, p). returns dict label-> (p, p_adj, reject@0.05)
    order=sorted(pairs, key=lambda x:x[1]); m=len(order); out={}; prev=0.0
    for i,(lab,p) in enumerate(order):
        adj=min(1.0, max(prev, (m-i)*p)); prev=adj
        out[lab]=(p, adj, adj<0.05)
    return out

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def eval_instance(n, edges, depth, budget):
    c_vals=qaoa_cost_values(edges,n); features=graph_features(n,edges)
    optcut_frac=float(c_vals.max()/max(len(edges),1))
    res={"_optcut_frac":optcut_frac,"_n":n,"_m":len(edges)}
    for m,(ancs,deltas) in methods_for(n,edges,features,depth).items():
        res[m]=refine_from_anchors(ancs,deltas,edges,n,c_vals,budget,depth)
    res["_features"]=features
    return res

def run_regime(rows, label, headline_baseline="anchors_fixed"):
    """rows: list of per-instance dicts (with method ratios). Single pool."""
    by={m:np.array([r[m] for r in rows]) for m in METHODS}
    print(f"\n===== {label}  (N={len(rows)}) =====")
    for m in METHODS:
        print(f"  {m:14s} mean={by[m].mean():.4f} median={np.median(by[m]):.4f} std={by[m].std():.4f}")
    out={"label":label,"N":len(rows),"means":{m:float(by[m].mean()) for m in METHODS}}
    comparisons={
        "uq_qaoa - anchors_fixed (uncertainty geometry, MATCHED anchors)":
            by["uq_qaoa"]-by["anchors_fixed"],
        "uq_qaoa - tqa_refine (vs pure physics)":
            by["uq_qaoa"]-by["tqa_refine"],
        "post_fixed - tqa_refine (learned CENTER vs physics)":
            by["post_fixed"]-by["tqa_refine"],
    }
    out["stats"]={}
    for name,d in comparisons.items():
        st=paired_stats(d); out["stats"][name]=st
        print(f"    Δ {name}:")
        print(f"        mean={st['mean']:+.4f} median={st['median']:+.4f} "
              f"win_rate={st['win_rate']:.2f} ({st['wins']}W/{st['losses']}L/{st['ties']}T) "
              f"d_z={st['cohens_dz']:+.2f} sign_p={st['sign_p']:.3g}")
    return out, by, comparisons

def main():
    depth=DEPTH
    budget=int(os.environ.get("OOD_BUDGET","18"))
    n_inst=int(os.environ.get("OOD_NINST","30"))
    sizes=[12,14,16]
    print(f"[cfg] depth={depth} budget={budget} n_inst/family={n_inst} sizes={sizes}")

    print("[setup] building library + GIN ...")
    lib_feats, feat_std = build_feature_reference(depth)
    _=predict_gaussian(8, core.graph_edges("ER",8,1), depth)  # force GIN train
    print("[setup] done.")

    # ---- in-distribution held-out distance distribution -> OOD threshold ----
    ind_rows=[]; ind_dists=[]
    for fam in IND_FAMILIES:
        for i in range(n_inst):
            sz=sizes[i%len(sizes)]
            edges=core.graph_edges(fam, sz, stable_seed("v3_ind", fam, i, sz))
            if not edges: continue
            r=eval_instance(sz, edges, depth, budget)
            r["_dist"]=nearest_train_dist(r["_features"], lib_feats, feat_std)
            ind_dists.append(r["_dist"]); ind_rows.append(r)
    thr=float(np.percentile(ind_dists, 95))
    print(f"[OOD threshold] 95th pct of in-dist held-out nearest-train dist = {thr:.3f} "
          f"(in-dist median {np.median(ind_dists):.3f})")

    # ---- OOD families ----
    fam_rows={}; fam_meta={}
    for fam,(gen,kind) in OOD_FAMILIES.items():
        rows=[]
        for i in range(n_inst):
            sz=sizes[i%len(sizes)]
            edges=gen(sz, stable_seed("v3_ood", fam, i, sz))
            if not edges: continue
            r=eval_instance(sz, edges, depth, budget)
            r["_dist"]=nearest_train_dist(r["_features"], lib_feats, feat_std)
            r["_is_ood"]=r["_dist"]>thr
            rows.append(r)
        fam_rows[fam]=rows
        frac_ood=np.mean([r["_is_ood"] for r in rows])
        med_opt=np.median([r["_optcut_frac"] for r in rows])
        fam_meta[fam]={"kind":kind,"frac_ood":float(frac_ood),
                       "med_optcut":float(med_opt),"n":len(rows)}
        print(f"[family {fam:11s} kind={kind:10s}] frac_genuine_OOD={frac_ood:.2f} "
              f"med_optcut_frac={med_opt:.2f} N={len(rows)}")

    # ---- in-distribution regime stats ----
    run_regime(ind_rows, "IN-DISTRIBUTION (trained families, held-out)")

    # ---- per-family ----
    print("\n########## PER-FAMILY (OOD candidates) ##########")
    per_family_out={}
    for fam in OOD_FAMILIES:
        out,_,_=run_regime(fam_rows[fam], f"family={fam} ({fam_meta[fam]['kind']}, "
                           f"fracOOD={fam_meta[fam]['frac_ood']:.2f})")
        per_family_out[fam]=out

    # ---- pooled GENUINE-OOD, excluding trivial (bipartite/wheel) ----
    nontrivial=[f for f in OOD_FAMILIES if OOD_FAMILIES[f][1]!="trivial"]
    genuine=[r for f in nontrivial for r in fam_rows[f] if r["_is_ood"]]
    print(f"\n########## POOLED GENUINE-OOD (non-trivial families, dist>thr): "
          f"N={len(genuine)} ##########")
    if genuine:
        out,by,comps=run_regime(genuine, "POOLED GENUINE-OOD (stratified stats below)")
        # stratified bootstrap + Holm across the 3 comparisons
        dby={name:{} for name in comps}
        for f in nontrivial:
            sub=[r for r in fam_rows[f] if r["_is_ood"]]
            if not sub: continue
            bb={m:np.array([r[m] for r in sub]) for m in METHODS}
            dby["uq_qaoa - anchors_fixed (uncertainty geometry, MATCHED anchors)"][f]=bb["uq_qaoa"]-bb["anchors_fixed"]
            dby["uq_qaoa - tqa_refine (vs pure physics)"][f]=bb["uq_qaoa"]-bb["tqa_refine"]
            dby["post_fixed - tqa_refine (learned CENTER vs physics)"][f]=bb["post_fixed"]-bb["tqa_refine"]
        ps=[]
        for name in comps:
            lo,hi=strat_bootstrap_ci(dby[name])
            st=paired_stats(comps[name]); ps.append((name, st["sign_p"]))
            print(f"    [stratified] Δ {name}: 95%CI[{lo:+.4f},{hi:+.4f}]")
        hh=holm(ps)
        print("    [Holm-corrected sign tests]")
        for name,(p,adj,rej) in hh.items():
            print(f"        {name[:55]:55s} p={p:.3g} p_holm={adj:.3g} reject={rej}")

    # ---- geometric probe pool ----
    geom=[f for f in OOD_FAMILIES if OOD_FAMILIES[f][1]=="geometric"]
    geom_rows=[r for f in geom for r in fam_rows[f]]
    print(f"\n########## GEOMETRIC PROBE (grid/tri/hex/rgg): N={len(geom_rows)} ##########")
    run_regime(geom_rows, "GEOMETRIC POOLED (does learned center beat physics here?)")

    # persist
    def _clean(r): return {k:(float(v) if isinstance(v,(int,float,np.floating)) else
                             (bool(v) if isinstance(v,(bool,np.bool_)) else None))
                           for k,v in r.items() if not k.startswith("_features")}
    payload={"cfg":{"depth":depth,"budget":budget,"n_inst":n_inst,"sizes":sizes,
                    "ood_threshold":thr},
             "family_meta":fam_meta,
             "in_distribution":[_clean(r) for r in ind_rows],
             "ood_by_family":{f:[_clean(r) for r in fam_rows[f]] for f in fam_rows}}
    op=pathlib.Path(os.environ.get("OOD_OUT","/tmp/ood_v3.json"))
    json.dump(payload, open(op,"w"), indent=2, default=float)
    print(f"\n[done] wrote {op}")

if __name__=="__main__":
    main()
