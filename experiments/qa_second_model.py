"""
Gap-closer 2: second answerer model (generality over the answering model).
Re-answers LoCoMo QA with llama3.2:3b (instead of qwen3:8b) over the SAME
qwen3-built store and SAME embedding retrieval (retrieval is model-independent),
for the three policies at k in {8,12}. Judged by qwen3:8b (judge held FIXED for
comparability with the main results). Tests whether 'dedup doesn't help'
replicates with a different answerer. Reuses caches; resumable; retry-hardened.
"""
import json, os, time, random
import numpy as np
import importlib.util
spec=importlib.util.spec_from_file_location("qd","experiments/qa_dedup_eval.py")
qd=importlib.util.module_from_spec(spec); spec.loader.exec_module(qd)
CACHE=qd.CACHE; ANSWERER="llama3.2:3b"; POLS=["full","naive_dedup","query_aware"]
KS=[int(x) for x in os.environ.get("KS","8").split(",")]   # single budget by default (tractable)
WORKCAP=int(os.environ.get("WORKCAP","360"))               # cap for a completable run

def _post_retry(path,body,timeout,tries=5):
    last=None
    for i in range(tries):
        try: return qd._post(path,body,timeout)
        except Exception as e: last=e; time.sleep(2*(i+1))
    raise last

def answer2(context,q):
    prompt=("Answer the question using ONLY the memory facts. Be concise; reply with just "
            "the answer. If unknown, say 'unknown'.\n\nMemory facts:\n"
            f"{context}\n\nQuestion: {q}\nAnswer:")
    out=_post_retry("/api/chat",{"model":ANSWERER,"messages":[{"role":"user","content":prompt}],
        "stream":False,"options":{"temperature":0,"seed":0}},300)
    return out["message"]["content"].strip()

def judge_qwen(question,gold,pred):
    prompt=(f"Question: {question}\nReference answer: {gold}\nCandidate answer: {pred}\n\n"
            "Does the candidate convey the same information as the reference answer? Ignore "
            'wording/format. Reply strictly JSON {"correct": true|false}.')
    out=_post_retry("/api/chat",{"model":qd.GEN,"messages":[{"role":"user","content":prompt}],
        "stream":False,"think":False,"format":"json","options":{"temperature":0,"seed":0}},300)
    try: return bool(json.loads(out["message"]["content"]).get("correct",False))
    except Exception: return False

def main():
    data=json.load(open(qd.DATA,encoding="utf-8"))
    ex=json.load(open(os.path.join(CACHE,"extractions.json"),encoding="utf-8"))
    embmap=json.load(open(os.path.join(CACHE,"embeddings.json"),encoding="utf-8"))
    qcache=json.load(open(os.path.join(CACHE,"q_embeddings.json"),encoding="utf-8"))
    af=os.path.join(CACHE,"qa_answers_llama32.json"); jf=os.path.join(CACHE,"qa_judge_llama32ans.json")
    ans=json.load(open(af,encoding="utf-8")) if os.path.exists(af) else {}
    jud=json.load(open(jf,encoding="utf-8")) if os.path.exists(jf) else {}
    rng=random.Random(qd.SEED)
    from collections import defaultdict
    # Build the full work list (deterministic), decoupled into two single-model phases
    work=[]  # (akey, gold, ctx)
    for ci,sample in enumerate(data):
        facts=qd.conv_store(ex[ci])
        E=np.asarray([embmap[f] for f in facts],dtype=np.float32); E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
        dd=qd.dedup_store(facts,E,qd.TAU_DEDUP); full_idx=list(range(len(facts)))
        qs=[q for q in sample["qa"] if str(q.get("category")) in qd.CATS and q.get("answer")]
        rng.shuffle(qs); qs=qs[:qd.QPER]
        for q in qs:
            question=q["question"]; gold=str(q["answer"])
            if question not in qcache: continue
            qv=np.asarray(qcache[question],dtype=np.float32); qv/=np.linalg.norm(qv)+1e-9
            for k in KS:
                for pol in POLS:
                    if pol=="full": sel=qd.topk_by_relevance(qv,E,full_idx,k)
                    elif pol=="naive_dedup": sel=qd.topk_by_relevance(qv,E,dd,k)
                    else: sel=qd.mmr(qv,E,full_idx,k,qd.LAMBDA)
                    ctx="\n".join(f"- {facts[i]}" for i in sel)
                    work.append((f"{ci}|{question}|{pol}|{k}",gold,ctx,question))
    work=work[:WORKCAP]

    # PHASE 1: all llama3.2 answers (single model loaded)
    t0=time.time(); done=0
    for akey,gold,ctx,question in work:
        if akey not in ans:
            ans[akey]=answer2(ctx,question); done+=1
            if done%50==0:
                json.dump(ans,open(af,"w",encoding="utf-8")); print(f"  [ans] {done} t={time.time()-t0:.0f}s",flush=True)
    json.dump(ans,open(af,"w",encoding="utf-8"))
    # PHASE 2: all qwen3 judgments (single model loaded)
    done=0
    for akey,gold,ctx,question in work:
        if akey not in jud:
            jud[akey]=judge_qwen(question,gold,ans[akey]); done+=1
            if done%50==0:
                json.dump(jud,open(jf,"w",encoding="utf-8")); print(f"  [judge] {done} t={time.time()-t0:.0f}s",flush=True)
    json.dump(jud,open(jf,"w",encoding="utf-8"))
    agg=defaultdict(lambda:[0,0])
    for akey,gold,ctx,question in work:
        parts=akey.split("|"); pol=parts[-2]; k=int(parts[-1])
        agg[(pol,k)][0]+=1 if jud.get(akey) else 0; agg[(pol,k)][1]+=1
    print("\n"+"="*60)
    print(f"SECOND ANSWERER = {ANSWERER} (judge=qwen3:8b, LoCoMo cats1-4, n=200)")
    print("="*60)
    print(f"{'policy':<13}"+"".join(f"k={k:>2}   " for k in KS))
    for pol in POLS:
        row=f"{pol:<13}"
        for k in KS:
            c,tot=agg[(pol,k)]; row+=f"{(c/tot if tot else 0):>6.3f} "
        print(row)
    print("\n(pattern check: naive_dedup dominated? query_aware ~ full?)")

if __name__=="__main__":
    main()
