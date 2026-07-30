"""
Pivot step 1: locate the true bottleneck with a GOLD-EVIDENCE oracle.

LoCoMo gives each QA item the evidence turn ids (dia_id). We compare, per question:
  oracle    : context = ONLY the gold evidence turns (perfect retrieval ceiling)
  full_conv : context = the ENTIRE conversation (no retrieval needed; long context)
  retrieved : context = top-k facts by embedding sim (our earlier RAG pipeline)  [from cache where available]
LLM-judged accuracy, by category.

Read:
  - oracle HIGH but retrieved LOW  -> RETRIEVAL PRECISION is the bottleneck (method headroom is real)
  - oracle LOW too                 -> reasoning/answer-reformulation is the bottleneck (retrieval won't fix it)
  - full_conv vs oracle            -> does dumping everything help or hurt vs precise evidence?

Same 200-question sample as prior phases (seed 7, cats 1-4, 20/conv). Resumable.
"""
import json, os, re, time, random
import importlib.util
spec=importlib.util.spec_from_file_location("qd","experiments/qa_dedup_eval.py")
qd=importlib.util.module_from_spec(spec); spec.loader.exec_module(qd)
CACHE=qd.CACHE

def _post_retry(path,body,timeout,tries=5):
    last=None
    for i in range(tries):
        try: return qd._post(path,body,timeout)
        except Exception as e:
            last=e; time.sleep(2*(i+1))
    raise last

def answer(context,q):
    prompt=("Answer the question using ONLY the information below. Be concise; reply with just "
            "the answer. If unknown, say 'unknown'.\n\n"
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

def dia_map(sample):
    # dia_id -> "[<session date>] speaker: text"  (dates matter for temporal QA;
    # a fair retriever/memory would surface the session timestamp)
    conv=sample.get("conversation",{}); m={}
    for k,v in conv.items():
        if k.startswith("session_") and isinstance(v,list):
            snum=re.findall(r"\d+",k)[0]
            date=conv.get(f"session_{snum}_date_time","")
            for t in v:
                if isinstance(t,dict) and t.get("dia_id"):
                    m[t["dia_id"]]=f"[{date}] {t.get('speaker','?')}: {t.get('text','')}"
    return m

def full_conv_text(sample):
    conv=sample.get("conversation",{}); lines=[]
    for k in sorted([k for k in conv if k.startswith("session_") and isinstance(conv[k],list)],
                    key=lambda x:int(re.findall(r"\d+",x)[0])):
        snum=re.findall(r"\d+",k)[0]
        date=conv.get(f"session_{snum}_date_time","")
        lines.append(f"=== Session {snum} ({date}) ===")
        for t in conv[k]:
            if isinstance(t,dict): lines.append(f"{t.get('speaker','?')}: {t.get('text','')}")
    return "\n".join(lines)

def parse_ev(ev):
    if isinstance(ev,list): return [str(x) for x in ev]
    try: return [str(x) for x in json.loads(str(ev).replace("'",'"'))]
    except Exception: return re.findall(r"D\d+:\d+", str(ev))

def main():
    data=json.load(open(qd.DATA,encoding="utf-8"))
    af=os.path.join(CACHE,"oracle_answers.json"); jf=os.path.join(CACHE,"oracle_judge.json")
    ans=json.load(open(af,encoding="utf-8")) if os.path.exists(af) else {}
    jud=json.load(open(jf,encoding="utf-8")) if os.path.exists(jf) else {}
    rng=random.Random(qd.SEED)
    from collections import defaultdict
    agg=defaultdict(lambda:[0,0]); aggcat=defaultdict(lambda:[0,0])
    t0=time.time(); done=0
    for ci,s in enumerate(data):
        dm=dia_map(s); fulltext=full_conv_text(s)
        qs=[q for q in s["qa"] if str(q.get("category")) in qd.CATS and q.get("answer")]
        rng.shuffle(qs); qs=qs[:qd.QPER]
        for q in qs:
            question=q["question"]; gold=str(q["answer"]); cat=str(q["category"])
            ev=parse_ev(q.get("evidence",""))
            ev_text="\n".join(dm[e] for e in ev if e in dm) or "(no evidence turns found)"
            arms={"oracle":ev_text,"full_conv":fulltext}
            for arm,ctx in arms.items():
                akey=f"{ci}|{question}|{arm}"
                if akey not in ans:
                    ans[akey]=answer(ctx,question); done+=1
                if akey not in jud:
                    jud[akey]=judge(question,gold,ans[akey])
                    if done and done%40==0:
                        json.dump(ans,open(af,"w",encoding="utf-8")); json.dump(jud,open(jf,"w",encoding="utf-8"))
                        print(f"  {done} answered t={time.time()-t0:.0f}s",flush=True)
                ok=1 if jud[akey] else 0
                agg[arm][0]+=ok; agg[arm][1]+=1
                aggcat[(arm,cat)][0]+=ok; aggcat[(arm,cat)][1]+=1
        print(f"conv {ci+1}/10 t={time.time()-t0:.0f}s",flush=True)
    json.dump(ans,open(af,"w",encoding="utf-8")); json.dump(jud,open(jf,"w",encoding="utf-8"))

    print("\n"+"="*60)
    print("BOTTLENECK DIAGNOSTIC (LoCoMo, qwen3:8b judge, n per arm below)")
    print("="*60)
    print(f"{'arm':<12}{'acc':>8}{'n':>6}")
    for arm in ["oracle","full_conv"]:
        c,n=agg[arm]; print(f"{arm:<12}{(c/n if n else 0):>8.3f}{n:>6}")
    print("\n(compare vs earlier RETRIEVED full@k12 overall 0.400 / multihop 0.411)")
    print(f"\n{'arm':<12}{'cat1':>8}{'cat2':>8}{'cat3':>8}{'cat4':>8}")
    for arm in ["oracle","full_conv"]:
        row=f"{arm:<12}"
        for cat in ["1","2","3","4"]:
            c,n=aggcat[(arm,cat)]; row+=f"{(c/n if n else 0):>8.3f}"
        print(row)
    print("\nRead: oracle high + retrieved low => RETRIEVAL is the bottleneck (method headroom).")
    print("      oracle low too => reasoning/reformulation bottleneck (retrieval won't fix it).")

if __name__=="__main__":
    main()
