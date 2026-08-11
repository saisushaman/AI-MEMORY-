"""
Gap-closer 3: second dataset (LongMemEval, oracle split). Tests whether the
paper's most distinctive finding -- temporal reasoning is a floor even with GOLD
evidence -- replicates beyond LoCoMo. The oracle split ships only the evidence
sessions per question; we answer from them with qwen3:8b (num_ctx=32768) and
LLM-judge, broken down by question_type. Stratified subsample for tractability.
Resumable; retry-hardened.
"""
import json, os, time, random
import importlib.util
spec=importlib.util.spec_from_file_location("qd","experiments/qa_dedup_eval.py")
qd=importlib.util.module_from_spec(spec); spec.loader.exec_module(qd)
CACHE=qd.CACHE
DATA="experiments/data/longmemeval_oracle.json"
PER_TYPE=40; SEED=13

def _post_retry(path,body,timeout,tries=5):
    last=None
    for i in range(tries):
        try: return qd._post(path,body,timeout)
        except Exception as e: last=e; time.sleep(2*(i+1))
    raise last

def answer(context,q):
    prompt=("Answer the question using ONLY the conversation(s) below. Be concise; reply with "
            "just the answer. If unknown, say 'unknown'.\n\n"
            f"{context}\n\nQuestion: {q}\nAnswer:")
    out=_post_retry("/api/chat",{"model":qd.GEN,"messages":[{"role":"user","content":prompt}],
        "stream":False,"think":False,"options":{"temperature":0,"seed":0,"num_ctx":32768}},900)
    return out["message"]["content"].strip()

def judge(question,gold,pred):
    prompt=(f"Question: {question}\nReference answer: {gold}\nCandidate answer: {pred}\n\n"
            "Does the candidate convey the same information as the reference answer? Ignore "
            'wording/format. Reply strictly JSON {"correct": true|false}.')
    out=_post_retry("/api/chat",{"model":qd.GEN,"messages":[{"role":"user","content":prompt}],
        "stream":False,"think":False,"format":"json","options":{"temperature":0,"seed":0}},300)
    try: return bool(json.loads(out["message"]["content"]).get("correct",False))
    except Exception: return False

def sessions_text(item):
    parts=[]
    dates=item.get("haystack_dates",[])
    for i,sess in enumerate(item.get("haystack_sessions",[])):
        d=dates[i] if i<len(dates) else ""
        parts.append(f"=== Session ({d}) ===")
        for turn in sess:
            if isinstance(turn,dict):
                parts.append(f"{turn.get('role','?')}: {turn.get('content','')}")
    return "\n".join(parts)

def main():
    data=json.load(open(DATA,encoding="utf-8"))
    # stratified subsample by question_type
    rng=random.Random(SEED)
    from collections import defaultdict
    bytype=defaultdict(list)
    for it in data: bytype[it.get("question_type")].append(it)
    sample=[]
    for t,items in bytype.items():
        rng.shuffle(items); sample.extend(items[:PER_TYPE])
    af=os.path.join(CACHE,"lme_answers.json"); jf=os.path.join(CACHE,"lme_judge.json")
    ans=json.load(open(af,encoding="utf-8")) if os.path.exists(af) else {}
    jud=json.load(open(jf,encoding="utf-8")) if os.path.exists(jf) else {}
    agg=defaultdict(lambda:[0,0]); t0=time.time(); done=0
    for it in sample:
        qid=it["question_id"]; question=it["question"]; gold=str(it["answer"]); qt=it["question_type"]
        ctx=sessions_text(it)
        if qid not in ans: ans[qid]=answer(ctx,question); done+=1
        if qid not in jud: jud[qid]=judge(question,gold,ans[qid])
        if done and done%20==0:
            json.dump(ans,open(af,"w",encoding="utf-8")); json.dump(jud,open(jf,"w",encoding="utf-8"))
            print(f"  {done} answered t={time.time()-t0:.0f}s",flush=True)
        ok=1 if jud[qid] else 0
        agg[qt][0]+=ok; agg[qt][1]+=1; agg["OVERALL"][0]+=ok; agg["OVERALL"][1]+=1
    json.dump(ans,open(af,"w",encoding="utf-8")); json.dump(jud,open(jf,"w",encoding="utf-8"))
    print("\n"+"="*64)
    print("LongMemEval (oracle split) -- ORACLE-answering accuracy by question type")
    print("qwen3:8b answerer + judge; gold evidence sessions in context")
    print("="*64)
    print(f"{'question_type':<28}{'acc':>8}{'n':>6}")
    for t in sorted(agg, key=lambda x:(x!='OVERALL', x)):
        c,n=agg[t]; print(f"{t:<28}{(c/n if n else 0):>8.3f}{n:>6}")
    print("\nRead: does temporal-reasoning sit BELOW other types even with gold")
    print("evidence? -> replicates the LoCoMo temporal-floor finding on a 2nd dataset.")

if __name__=="__main__":
    main()
