#!/usr/bin/env python3
"""Evaluate a cached, test-independent LLM-guided adaptive cAL policy."""
from __future__ import annotations
import importlib.util, json, pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "clinrisk_agent"
OUT.mkdir(parents=True, exist_ok=True)
POLICY = json.load(open(Path(__file__).with_name("clinrisk_llm_policy.json")))
N_SPLITS, TEST_FRAC, INITIAL_N, BATCH, CANDIDATES = 60, .30, 10, 5, 25
BUDGET_POINTS = tuple(range(10, 101, 10))


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def split(y, seed):
    rng=np.random.RandomState(seed+1000); pool=[]; test=[]
    for c in (0,1):
        z=np.flatnonzero(y==c); rng.shuffle(z); k=max(1,round(.3*len(z)))
        test.extend(z[:k]); pool.extend(z[k:])
    return np.array(pool),np.array(test)


def gamma(X): return 1/(X.shape[1]*X.var()) if X.var() else 1.


def met(y,p,s,r,a,nlab):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {"fnr":fn/(fn+tp),"specificity":tn/(tn+fp),"precision":precision_score(y,p,zero_division=0),
            "balanced_accuracy":balanced_accuracy_score(y,p),"auc":roc_auc_score(y,s),
            "cost4":100*(4*fn+fp)/len(y),"fn":int(fn),"fp":int(fp),
            "risk_weight":r,"query_priority":a,"n_labeled":nlab}


def oof_context(X,y):
    counts=np.bincount(y,minlength=2)
    if counts.min()<3: return None,None
    cv=StratifiedKFold(3,shuffle=True,random_state=len(y))
    clf=SVC(C=1,kernel="rbf",gamma=gamma(X),class_weight={0:1,1:1})
    pred=cross_val_predict(clf,X,y,cv=cv,method="predict")
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return tp/(tp+fn) if tp+fn else None, tn/(tn+fp) if tn+fp else None


def smooth(new,old): return int(np.clip(new,old-1,old+1))


def llm_policy(ctx, prev_r, prev_a):
    sens,spec=ctx["sens"],ctx["spec"]
    if sens is None or spec is None: nr,na=2,2
    elif spec<.50: return 1,4,"specificity_guard"
    else:
        if spec<.60: nr=min(prev_r,2)
        elif spec>=.70 and sens<.60: nr=4
        elif spec>=.70 and sens<.75: nr=3
        elif spec>=.70 and ctx["trend"]=="declining" and ctx["margin"]>=.10: nr=min(prev_r+1,4)
        else: nr=prev_r
        if spec<.60: na=3
        elif spec>=.70 and sens<.75 and ctx["neg_frac"]>=.85: na=3
        elif ctx["margin"]<.10 and spec>=.60: na=max(prev_a,3)
        elif ctx["prevalence"]<.15 or ctx["prevalence"]>.85: na=max(prev_a,2)
        else: na=prev_a
    nr,na=smooth(nr,prev_r),smooth(na,prev_a)
    if ctx["budget"]>=.85: nr=prev_r
    return nr,na,"llm_distilled"


def rule_policy(ctx,prev_r,prev_a):
    sens,spec=ctx["sens"],ctx["spec"]
    if sens is None or spec is None: return 2,2,"cold_start"
    if spec<.50: return 1,3,"specificity_guard"
    if sens<.65 and spec>=.70: return smooth(4,prev_r),smooth(3,prev_a),"recall_escalation"
    if sens<.80 and spec>=.65: return smooth(3,prev_r),smooth(2,prev_a),"moderate_escalation"
    return smooth(2,prev_r),smooth(2,prev_a),"balanced_hold"


def query(clf,X,a,rng):
    d=clf.decision_function(X); score=np.where(d<=0,np.abs(d)/a,np.abs(d))
    cand=np.argsort(score)[:min(CANDIDATES,len(X))]
    # Lightweight diversity: select across quantiles of the uncertainty-ranked candidate list.
    if len(cand)<=BATCH:return cand
    positions=np.linspace(0,len(cand)-1,BATCH).round().astype(int)
    return cand[positions]


def run_one(Xp,yp,Xt,yt,seed,mode):
    rng=np.random.RandomState(seed); order=list(rng.permutation(len(yp)))
    lab=order[:INITIAL_N]; unlab=order[INITIAL_N:]
    while len(set(yp[lab]))<2: lab.append(unlab.pop())
    r,a=1,1; sens_hist=[]; rows=[]; next_points=list(BUDGET_POINTS)
    while True:
        frac=len(lab)/len(yp)
        sens,spec=oof_context(Xp[lab],yp[lab])
        cw={0:1,1:r}; clf=SVC(C=1,kernel="rbf",gamma=gamma(Xp),class_weight=cw).fit(Xp[lab],yp[lab])
        du=clf.decision_function(Xp[unlab]) if unlab else np.array([])
        trend="flat"
        if sens is not None and len(sens_hist)>=2:
            trend="declining" if sens<sens_hist[-1]-.02 else ("improving" if sens>sens_hist[-1]+.02 else "flat")
        if sens is not None:sens_hist.append(sens)
        ctx={"budget":frac,"sens":sens,"spec":spec,"prevalence":yp[lab].mean(),
             "margin":float(np.median(np.abs(du))) if len(du) else 0.,"neg_frac":float((du<=0).mean()) if len(du) else 0.,"trend":trend}
        if mode=="standard": nr,na,why=1,1,"fixed"
        elif mode=="fixed_r4": nr,na,why=4,4,"fixed"
        elif mode=="adaptive_rule": nr,na,why=rule_policy(ctx,r,a)
        else: nr,na,why=llm_policy(ctx,r,a)
        r,a=nr,na
        clf=SVC(C=1,kernel="rbf",gamma=gamma(Xp),class_weight={0:1,1:r}).fit(Xp[lab],yp[lab])
        score=clf.decision_function(Xt); pred=(score>0).astype(int)
        while next_points and frac*100>=next_points[0]-1:
            rows.append({"budget":next_points.pop(0),"mode":mode,"reason":why,**met(yt,pred,score,r,a,len(lab))})
        if not unlab: break
        local=query(clf,Xp[unlab],a,rng); chosen=[unlab[i] for i in local]
        lab.extend(chosen); chosen=set(chosen); unlab=[i for i in unlab if i not in chosen]
    return rows


def main():
    gait=module("gait",ROOT/"experiments"/"gait_al_simulation.py"); gait.DATA_DIR=ROOT/"datasets"/"gait"/"physionet-vgrf"; gait.DEMO_PATH=gait.DATA_DIR/"demographics.txt"
    speech=module("speech",ROOT/"experiments"/"speech_al_simulation.py"); speech.DATA_PATH=ROOT/"datasets"/"pd_speech_features.csv"
    Xg,yg,_,_=gait.load_gait_features(); Xs,ys,_=speech.load_patient_features(speech.DATA_PATH)
    summary={}
    for name,X,y in (("gait",Xg,yg),("speech",Xs,ys)):
        allrows=[]
        for seed in range(N_SPLITS):
            po,te=split(y,seed); sc=StandardScaler().fit(X[po]); Xp,Xt=sc.transform(X[po]),sc.transform(X[te])
            for mode in ("standard","fixed_r4","adaptive_rule","clinrisk_llm"):
                allrows.extend({"seed":seed,**z} for z in run_one(Xp,y[po],Xt,y[te],seed,mode))
        with open(OUT/f"{name}_records.pkl","wb") as f:pickle.dump(allrows,f)
        summary[name]={}
        for mode in ("standard","fixed_r4","adaptive_rule","clinrisk_llm"):
            q=[z for z in allrows if z["mode"]==mode and z["budget"]==50]
            summary[name][mode]={k:{"mean":float(np.mean([z[k] for z in q])),"std":float(np.std([z[k] for z in q]))} for k in ("cost4","fnr","specificity","balanced_accuracy","risk_weight")}
        print(name,summary[name],flush=True)
    json.dump(summary,open(OUT/"summary.json","w"),indent=2)

if __name__=="__main__":main()
