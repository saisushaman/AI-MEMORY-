"""
Phase-3e: re-score the Phase-3d budget-sweep answers with an LLM judge (the
metric LoCoMo is designed for), broken down by category. String-F1/containment
structurally undercounts LoCoMo (answers are reformulated/normalized), which can
mask real differences between policies. This reuses ALL cached answers (no new
extraction or answering) and only adds judge calls. Resumable (judgments cached).

Hypothesis to test fairly: on MULTI-HOP (category 1), where the answer needs
coverage of several distinct facts, does query_aware (MMR) beat full at matched
budget k -- and is naive_dedup dominated? Single-hop (cat 4) as contrast.
"""
import json, os, time
import importlib.util
spec=importlib.util.spec_from_file_location("qd","experiments/qa_dedup_eval.py")
qd=importlib.util.module_from_spec(spec); spec.loader.exec_module(qd)
CACHE=qd.CACHE
CATNAME={"1":"multihop","2":"temporal","3":"open","4":"single"}

def judge(question,gold,pred):
    prompt=(f"Question: {question}\nReference answer: {gold}\nCandidate answer: {pred}\n\n"
            "Does the candidate answer convey the same information as the reference "
            "answer to the question? Ignore wording/format differences. "
            'Reply strictly as JSON: {"correct": true} or {"correct": false}.')
    out=qd._post("/api/chat",{"model":qd.GEN,"messages":[{"role":"user","content":prompt}],
        "stream":False,"think":False,"format":"json","options":{"temperature":0,"seed":0}},300)
    try: return bool(json.loads(out["message"]["content"]).get("correct",False))
    except Exception: return False

def main():
    data=json.load(open(qd.DATA,encoding="utf-8"))
    ans=json.load(open(os.path.join(CACHE,"qa_answers_sweep.json"),encoding="utf-8"))
    # (ci,question) -> (gold,category)
    meta={}
    for ci,s in enumerate(data):
        for q in s["qa"]:
            if q.get("answer"): meta[(ci,q["question"])]=(str(q["answer"]),str(q.get("category")))
    jf=os.path.join(CACHE,"judgments.json")
    jud=json.load(open(jf,encoding="utf-8")) if os.path.exists(jf) else {}

    keys=[k for k in ans if k.count("|")>=3]
    t0=time.time(); done=0
    for key in keys:
        if key in jud: continue
        parts=key.split("|"); ci=int(parts[0]); k_=parts[-1]; pol=parts[-2]
        question="|".join(parts[1:-2])
        m=meta.get((ci,question))
        if not m: continue
        gold,_=m
        jud[key]=judge(question,gold,ans[key]); done+=1
        if done%100==0:
            json.dump(jud,open(jf,"w",encoding="utf-8"))
            print(f"  judged {done} t={time.time()-t0:.0f}s",flush=True)
    json.dump(jud,open(jf,"w",encoding="utf-8"))

    # aggregate: (policy,k) overall + by category
    from collections import defaultdict
    agg=defaultdict(lambda:[0,0])          # (pol,k)->[correct,n]
    aggcat=defaultdict(lambda:[0,0])       # (pol,k,cat)->[correct,n]
    for key,ok in jud.items():
        parts=key.split("|"); ci=int(parts[0]); k_=parts[-1]; pol=parts[-2]
        question="|".join(parts[1:-2]); m=meta.get((ci,question))
        if not m: continue
        cat=m[1]
        agg[(pol,k_)][0]+=ok; agg[(pol,k_)][1]+=1
        aggcat[(pol,k_,cat)][0]+=ok; aggcat[(pol,k_,cat)][1]+=1

    KS=["2","4","8","12"]; pols=["full","naive_dedup","query_aware"]
    print("\n"+"="*70)
    print("LLM-JUDGE accuracy (LoCoMo, qwen3:8b judge) -- OVERALL by policy x k")
    print("="*70)
    print(f"{'policy':<13}"+"".join(f"k={k:>2}   " for k in KS))
    for pol in pols:
        row=f"{pol:<13}"
        for k in KS:
            c,n=agg[(pol,k)]; row+=f"{(c/n if n else 0):>6.3f} "
        print(row)
    for cat in ["1","4"]:
        print(f"\n--- category {cat} ({CATNAME[cat]}) ---")
        print(f"{'policy':<13}"+"".join(f"k={k:>2}   " for k in KS))
        for pol in pols:
            row=f"{pol:<13}"
            for k in KS:
                c,n=aggcat[(pol,k,cat)]; row+=f"{(c/n if n else 0):>6.3f} "
            n0=aggcat[(pol,KS[0],cat)][1]; print(row+f"  (n={n0})")
    print("\nRead: on cat 1 (multihop), does query_aware beat full at matched k? Is naive_dedup dominated?")

if __name__=="__main__":
    main()
