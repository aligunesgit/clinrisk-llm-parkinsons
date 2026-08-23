#!/usr/bin/env python3
from pathlib import Path
import pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"results"/"clinrisk_agent"; F=ROOT/"manuscripts"/"figures"
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"legend.fontsize":7.5,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
MODES=["standard","fixed_r4","adaptive_rule","clinrisk_llm"]
LAB={"standard":"Standard AL","fixed_r4":"Fixed cAL ($r=4$)","adaptive_rule":"Adaptive Rule","clinrisk_llm":"ClinRisk-LLM"}
COL={"standard":"#4C72B0","fixed_r4":"#8172B2","adaptive_rule":"#55A868","clinrisk_llm":"#C44E52"}
def load(n): return pickle.load(open(R/f"{n}_records.pkl","rb"))
def v(rows,m,b,k):return np.array([x[k] for x in rows if x["mode"]==m and x["budget"]==b],float)
def main():
 g,s=load("gait"),load("speech")
 fig,ax=plt.subplots(2,3,figsize=(7.1,4.8),sharex=True)
 for i,(rows,name) in enumerate(((g,"Gait"),(s,"Speech"))):
  for j,(key,title) in enumerate((("cost4",r"Common cost $C_4$ (points per 100)"),("fnr","False-negative rate"),("specificity","Specificity"))):
   for m in MODES:
    bs=np.arange(10,101,10); mean=[]; sem=[]
    for b in bs:
     z=v(rows,m,b,key);mean.append(z.mean());sem.append(z.std(ddof=1)/np.sqrt(len(z)))
    mean=np.array(mean);sem=np.array(sem)
    ax[i,j].plot(bs,mean,color=COL[m],marker="o",ms=2.8,lw=1.4,label=LAB[m]);ax[i,j].fill_between(bs,mean-sem,mean+sem,color=COL[m],alpha=.11)
   ax[i,j].grid(alpha=.22,lw=.5);ax[i,j].set_title(title);ax[i,j].set_xticks([10,25,50,75,100])
   if i==1:ax[i,j].set_xlabel("Pool labeled (\%)")
  ax[i,0].set_ylabel(name)
 h,l=ax[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=4,frameon=False,loc="lower center",bbox_to_anchor=(.5,-.01));fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(F/"fig_clinrisk_learning_curves.pdf",bbox_inches="tight");plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(7,3.1),sharey=True)
 for a,(rows,name) in zip(ax,((g,"(a) Gait"),(s,"(b) Speech"))):
  for m in ("adaptive_rule","clinrisk_llm"):
   bs=np.arange(10,101,10);mean=[v(rows,m,b,"risk_weight").mean() for b in bs];sd=[v(rows,m,b,"risk_weight").std() for b in bs]
   a.plot(bs,mean,color=COL[m],marker="o",lw=1.5,label=LAB[m]);a.fill_between(bs,np.array(mean)-np.array(sd),np.array(mean)+np.array(sd),color=COL[m],alpha=.12)
  a.set_title(name);a.set_xlabel("Pool labeled (\%)");a.set_ylim(.8,4.2);a.set_yticks([1,2,3,4]);a.grid(alpha=.22,lw=.5)
 ax[0].set_ylabel("Selected risk weight $r_t$");h,l=ax[0].get_legend_handles_labels();fig.legend(h,l,ncol=2,frameon=False,loc="lower center",bbox_to_anchor=(.5,-.07));fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(F/"fig_clinrisk_risk_trajectory.pdf",bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()
