"""
Phase-3d: budget sweep. Tests the EFFICIENCY half missing from Phase-3c by
varying the retrieval budget k and plotting accuracy vs context tokens for the
three memory policies. The thesis predicts query_aware reaches full's accuracy
at smaller k (fewer tokens), while naive_dedup is dominated everywhere.

Reuses Phase-3c helpers, question embeddings, and the k=8 answer cache
(pre-seeded). Answer cache keyed by (conv|question|policy|k). Resumable.
Same 200-question sample as Phase-3c (identical seed/sampling).
"""
import json, os, time, random
import numpy as np
import importlib.util

spec=importlib.util.spec_from_file_location("qd","experiments/qa_dedup_eval.py")
qd=importlib.util.module_from_spec(spec); spec.loader.exec_module(qd)

KS=[2,4,8,12]
CACHE=qd.CACHE

def main():
    data=json.load(open(qd.DATA,encoding="utf-8"))
    ex=json.load(open(os.path.join(CACHE,"extractions.json"),encoding="utf-8"))
    embmap=json.load(open(os.path.join(CACHE,"embeddings.json"),encoding="utf-8"))
    qcache_f=os.path.join(CACHE,"q_embeddings.json")
    qcache=json.load(open(qcache_f,encoding="utf-8")) if os.path.exists(qcache_f) else {}

    # pre-seed k=8 answers from Phase-3c cache (keys "ci|q|pol" -> add "|8")
    acache_f=os.path.join(CACHE,"qa_answers_sweep.json")
    acache=json.load(open(acache_f,encoding="utf-8")) if os.path.exists(acache_f) else {}
    old_f=os.path.join(CACHE,"qa_answers.json")
    if os.path.exists(old_f):
        for k,v in json.load(open(old_f,encoding="utf-8")).items():
            nk=k+"|8"
            if nk not in acache: acache[nk]=v

    rng=random.Random(qd.SEED)
    policies=["full","naive_dedup","query_aware"]
    agg={(p,k):{"f1":0.0,"contain":0.0,"ctx":0,"n":0} for p in policies for k in KS}
    t0=time.time(); done=0
    for ci,sample in enumerate(data):
        facts=qd.conv_store(ex[ci])
        E=np.asarray([embmap[f] for f in facts],dtype=np.float32); E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
        dd_idx=qd.dedup_store(facts,E,qd.TAU_DEDUP); full_idx=list(range(len(facts)))
        qs=[q for q in sample["qa"] if str(q.get("category")) in qd.CATS and q.get("answer")]
        rng.shuffle(qs); qs=qs[:qd.QPER]
        for q in qs:
            question=q["question"]; gold=str(q["answer"])
            if question not in qcache: qcache[question]=qd.embed(question)
            qv=np.asarray(qcache[question],dtype=np.float32); qv/=np.linalg.norm(qv)+1e-9
            for k in KS:
                for pol in policies:
                    if pol=="full": sel=qd.topk_by_relevance(qv,E,full_idx,k)
                    elif pol=="naive_dedup": sel=qd.topk_by_relevance(qv,E,dd_idx,k)
                    else: sel=qd.mmr(qv,E,full_idx,k,qd.LAMBDA)
                    ctx="\n".join(f"- {facts[i]}" for i in sel)
                    ctx_tok=sum(qd.tok(facts[i]) for i in sel)
                    akey=f"{ci}|{question}|{pol}|{k}"
                    if akey not in acache:
                        acache[akey]=qd.answer(ctx,question); done+=1
                        if done%50==0:
                            json.dump(acache,open(acache_f,"w",encoding="utf-8"))
                            json.dump(qcache,open(qcache_f,"w",encoding="utf-8"))
                            print(f"  {done} new answers t={time.time()-t0:.0f}s",flush=True)
                    pred=acache[akey]
                    a=agg[(pol,k)]; a["f1"]+=qd.f1(pred,gold); a["contain"]+=qd.contains(pred,gold)
                    a["ctx"]+=ctx_tok; a["n"]+=1
        print(f"conv {ci+1}/10 t={time.time()-t0:.0f}s",flush=True)
    json.dump(acache,open(acache_f,"w",encoding="utf-8")); json.dump(qcache,open(qcache_f,"w",encoding="utf-8"))

    print("\n"+"="*72)
    print("BUDGET SWEEP: accuracy vs context tokens by policy (LoCoMo cats1-4, qwen3:8b)")
    print("="*72)
    print(f"{'policy':<13}{'k':>4}{'F1':>8}{'contain':>9}{'ctx_tok/q':>11}")
    for pol in policies:
        for k in KS:
            a=agg[(pol,k)]; n=max(1,a["n"])
            print(f"{pol:<13}{k:>4}{a['f1']/n:>8.3f}{a['contain']/n:>9.3f}{a['ctx']/n:>11.1f}")
    print("\nPareto read: at matched F1, which policy uses fewer ctx tokens? "
          "Does query_aware reach full's best F1 at smaller k?")

if __name__=="__main__":
    main()
