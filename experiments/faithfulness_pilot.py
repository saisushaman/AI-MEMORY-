"""
Direction-3 phenomenon check: what fraction of the LLM-extracted memory store is
UNSUPPORTED by its source conversation (i.e. hallucinated by the extractor)?

Reuses the Phase-3b store (experiments/cache/extractions.json, built by qwen3:8b
from LoCoMo). For a sample of extracted facts, an LLM judge checks whether the
fact is fully supported by the SOURCE SESSION it was extracted from. Reports the
unsupported (hallucination) rate. Cheap 'is there a phenomenon' probe, like the
duplication pilot. Local Ollama only; retry-hardened; resumable.
"""
import json, os, re, time, random
import importlib.util
spec=importlib.util.spec_from_file_location("qd","experiments/qa_dedup_eval.py")
qd=importlib.util.module_from_spec(spec); spec.loader.exec_module(qd)
CACHE=qd.CACHE
SAMPLE_PER_CONV=40
SEED=11

def _post_retry(path,body,timeout,tries=5):
    last=None
    for i in range(tries):
        try: return qd._post(path,body,timeout)
        except Exception as e: last=e; time.sleep(2*(i+1))
    raise last

def session_texts(sample):
    conv=sample.get("conversation",{}); out=[]
    for k in sorted([k for k in conv if k.startswith("session_") and isinstance(conv[k],list)],
                    key=lambda x:int(re.findall(r"\d+",x)[0])):
        lines=[f"{t.get('speaker','?')}: {t.get('text','')}" for t in conv[k] if isinstance(t,dict)]
        out.append("\n".join(lines))
    return out

def check_support(session_text,fact):
    prompt=("You are a strict fact-checker. Given the DIALOGUE, decide whether the STATEMENT is "
            "fully supported by (directly stated or unambiguously entailed by) the dialogue. "
            "If any part is not supported or is invented, it is not supported.\n\n"
            f"DIALOGUE:\n{session_text}\n\nSTATEMENT: {fact}\n\n"
            'Reply strictly JSON: {"supported": true} or {"supported": false}.')
    out=_post_retry("/api/chat",{"model":qd.GEN,"messages":[{"role":"user","content":prompt}],
        "stream":False,"think":False,"format":"json","options":{"temperature":0,"seed":0,"num_ctx":8192}},300)
    try: return bool(json.loads(out["message"]["content"]).get("supported",True))
    except Exception: return True   # conservative: assume supported on parse failure (under-counts hallucination)

def main():
    data=json.load(open(qd.DATA,encoding="utf-8"))
    ex=json.load(open(os.path.join(CACHE,"extractions.json"),encoding="utf-8"))
    jf=os.path.join(CACHE,"faithfulness_judge.json")
    jud=json.load(open(jf,encoding="utf-8")) if os.path.exists(jf) else {}
    rng=random.Random(SEED)
    total=0; unsupported=0; examples=[]; t0=time.time(); done=0
    for ci,sample in enumerate(data):
        sess=session_texts(sample)
        # collect (session_idx, fact) pairs for this conv
        pairs=[]
        for si,facts in enumerate(ex[ci]):
            if si>=len(sess): continue
            for f in facts:
                pairs.append((si,f))
        rng.shuffle(pairs); pairs=pairs[:SAMPLE_PER_CONV]
        for si,f in pairs:
            key=f"{ci}|{si}|{f}"
            if key not in jud:
                jud[key]=check_support(sess[si],f); done+=1
                if done%40==0:
                    json.dump(jud,open(jf,"w",encoding="utf-8"))
                    print(f"  checked {done} t={time.time()-t0:.0f}s",flush=True)
            sup=jud[key]; total+=1
            if not sup:
                unsupported+=1
                if len(examples)<12: examples.append(f)
        print(f"conv {ci+1}/10 t={time.time()-t0:.0f}s",flush=True)
    json.dump(jud,open(jf,"w",encoding="utf-8"))
    print("\n"+"="*64)
    print("EXTRACTOR FAITHFULNESS (LoCoMo store built by qwen3:8b; qwen3 judge)")
    print("="*64)
    print(f"facts checked: {total}")
    print(f"unsupported (hallucinated) : {unsupported}  = {unsupported/max(1,total):.1%}")
    print(f"supported                  : {total-unsupported}  = {(total-unsupported)/max(1,total):.1%}")
    print("\nExample UNSUPPORTED extracted 'memories':")
    for e in examples: print("  -", e[:100])
    print("\nRead: high unsupported rate => the LLM-built memory store contains hallucinated")
    print("facts (a real phenomenon worth studying). ~0 => direction is dead.")

if __name__=="__main__":
    main()
