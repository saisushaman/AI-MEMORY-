"""
Gap-closer 1: judge robustness + finding-generality under a second judge.
Re-judges the EXISTING cached qwen3 sweep answers with a DIFFERENT judge model
(llama3.2:3b). Reports (a) inter-judge agreement (raw + Cohen's kappa) vs the
qwen3 judge, and (b) the budget-sweep accuracy table under the llama3.2 judge
(does 'dedup doesn't help' survive a different judge?). No new answer generation.
Resumable; retry-hardened.
"""
import json, os, time
import importlib.util
spec=importlib.util.spec_from_file_location("qd","experiments/qa_dedup_eval.py")
qd=importlib.util.module_from_spec(spec); spec.loader.exec_module(qd)
CACHE=qd.CACHE; JUDGE2="llama3.2:3b"; KS=["2","4","8","12"]; POLS=["full","naive_dedup","query_aware"]

def _post_retry(path,body,tries=5):
    last=None
    for i in range(tries):
        try: return qd._post(path,body,300)
        except Exception as e: last=e; time.sleep(2*(i+1))
    raise last

def judge2(question,gold,pred):
    prompt=(f"Question: {question}\nReference answer: {gold}\nCandidate answer: {pred}\n\n"
            "Does the candidate convey the same information as the reference answer? Ignore "
            'wording/format. Reply strictly JSON {"correct": true|false}.')
    out=_post_retry("/api/chat",{"model":JUDGE2,"messages":[{"role":"user","content":prompt}],
        "stream":False,"format":"json","options":{"temperature":0,"seed":0}})
    try: return bool(json.loads(out["message"]["content"]).get("correct",False))
    except Exception: return False

def main():
    data=json.load(open(qd.DATA,encoding="utf-8"))
    ans=json.load(open(os.path.join(CACHE,"qa_answers_sweep.json"),encoding="utf-8"))
    j1=json.load(open(os.path.join(CACHE,"judgments.json"),encoding="utf-8"))   # qwen3 judge
    j2f=os.path.join(CACHE,"judgments_llama32.json")
    j2=json.load(open(j2f,encoding="utf-8")) if os.path.exists(j2f) else {}
    meta={}
    for ci,s in enumerate(data):
        for q in s["qa"]:
            if q.get("answer"): meta[(ci,q["question"])]=str(q["answer"])

    keys=[k for k in ans if k in j1]   # only judge what qwen3 already judged (aligned set)
    t0=time.time(); done=0
    for key in keys:
        if key in j2: continue
        parts=key.split("|"); ci=int(parts[0]); question="|".join(parts[1:-2])
        gold=meta.get((ci,question))
        if gold is None: continue
        j2[key]=judge2(question,gold,ans[key]); done+=1
        if done%100==0:
            json.dump(j2,open(j2f,"w",encoding="utf-8")); print(f"  judged {done} t={time.time()-t0:.0f}s",flush=True)
    json.dump(j2,open(j2f,"w",encoding="utf-8"))

    # agreement over the aligned set
    both=[k for k in keys if k in j1 and k in j2]
    agree=sum(1 for k in both if j1[k]==j2[k]); n=len(both)
    p_obs=agree/n
    # Cohen's kappa
    a1=sum(1 for k in both if j1[k])/n; a2=sum(1 for k in both if j2[k])/n
    p_exp=a1*a2+(1-a1)*(1-a2); kappa=(p_obs-p_exp)/(1-p_exp) if p_exp<1 else 1.0
    print("\n"+"="*60)
    print(f"JUDGE AGREEMENT qwen3 vs llama3.2 (n={n} answers)")
    print("="*60)
    print(f"raw agreement: {p_obs:.3f} | Cohen's kappa: {kappa:.3f}")
    print(f"positive rate: qwen3={a1:.3f}  llama3.2={a2:.3f}")

    # sweep accuracy under llama3.2 judge (cats 1-4)
    from collections import defaultdict
    agg=defaultdict(lambda:[0,0])
    for key in both:
        parts=key.split("|"); pol=parts[-2]; k=parts[-1]
        agg[(pol,k)][0]+=1 if j2[key] else 0; agg[(pol,k)][1]+=1
    print(f"\nBudget sweep under llama3.2 judge:")
    print(f"{'policy':<13}"+"".join(f"k={k:>2}   " for k in KS))
    for pol in POLS:
        row=f"{pol:<13}"
        for k in KS:
            c,tot=agg[(pol,k)]; row+=f"{(c/tot if tot else 0):>6.3f} "
        print(row)
    print("\n(compare pattern to Table: dedup should still be dominated / query-aware ~ full)")

if __name__=="__main__":
    main()
