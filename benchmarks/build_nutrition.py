"""Rebuild a Russian-nutrition retrieval benchmark → the cube schema.

The original nutrition.jsonl was lost (systemd-tmpfiles swept the /tmp datadir 2026-07-22) and its
builder was never committed. This constructs a NEW, principled set from the surviving corpus in the
nutribot DB: the `docs/` kind of `nrag_chunks` = Russian nutrition-lecture PDFs ("Неженская — Лекции").
It is NOT the original set (numbers are not comparable to prior published nutrition results).

Method (standard passage-retrieval construction): pick well-spaced substantive chunks as GOLD; for each,
have an LLM write ONE specific Russian question whose answer is contained in that chunk; the item's `docs`
= the gold chunk + hard distractors sampled from elsewhere in the corpus. gold = [that chunk_id].

    CORPUS=nutrition_corpus.jsonl OUT=benchmarks/datasets/nutrition.jsonl TARGET=120 python benchmarks/build_nutrition.py

Source content is third-party copyrighted (proprietary), used internally only; not redistributed.
"""
import os, re, json, hashlib, sys
from openai import OpenAI

CORPUS = os.environ["CORPUS"]
OUT = os.environ.get("OUT", "benchmarks/datasets/nutrition.jsonl")
TARGET = int(os.environ.get("TARGET", "120"))
NDOCS = int(os.environ.get("NDOCS", "20"))           # docs per item (gold + distractors)
QWEN = os.environ.get("QWEN_URL", "http://192.168.40.105:30807/v1")
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
cli = OpenAI(base_url=QWEN, api_key="EMPTY")

def norm(t):
    return re.sub(r"[ \t]{2,}", " ", t.replace("\xa0", " ")).strip()

def h(*xs):
    return hashlib.md5("|".join(map(str, xs)).encode()).hexdigest()

rows = [json.loads(l) for l in open(CORPUS)]
for i, r in enumerate(rows):
    r["text"] = norm(r["text"]); r["cid"] = "n%04d" % i           # stable unique chunk id
rows = [r for r in rows if len(r["text"]) >= 500]                 # substantive only
# deterministic order, then stride so gold chunks are far apart (reduce adjacent-chunk answer overlap)
rows.sort(key=lambda r: h(r["cid"]))
N = len(rows)
stride = max(1, N // (TARGET + 40))
cand = rows[::stride]

SYS = ("Ты — составитель бенчмарка по нутрициологии. По данному фрагменту лекции сформулируй ОДИН "
       "конкретный фактический вопрос на русском языке, ответ на который ОДНОЗНАЧНО содержится в этом "
       "фрагменте (не общий, а именно по фактам фрагмента). Дай также краткий ответ (несколько слов). "
       'Верни СТРОГО JSON без пояснений: {"question": "...", "answer": "..."}')

def gen(text):
    for _ in range(2):
        try:
            r = cli.chat.completions.create(model="Qwen3.6-35B-A3B", temperature=0.2, max_tokens=300,
                extra_body=NOTHINK, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": text[:2800]}])
            o = json.loads(r.choices[0].message.content)
            q, a = (o.get("question") or "").strip(), (o.get("answer") or "").strip()
            if len(q) >= 12 and re.search(r"[а-яА-Я]", q) and a:
                return q, a
        except Exception as e:
            print("  gen err:", str(e)[:80], flush=True)
    return None, None

items, used = [], 0
for c in cand:
    if len(items) >= TARGET:
        break
    q, a = gen(c["text"])
    used += 1
    if not q:
        continue
    # distractors: sample from chunks far from this one in the hashed order (avoid same-topic neighbours)
    pool = [r for r in rows if r["cid"] != c["cid"]]
    pool.sort(key=lambda r: h(c["cid"], r["cid"]))
    docs = [{"chunk_id": c["cid"], "text": c["text"], "created_at": None}] + \
           [{"chunk_id": d["cid"], "text": d["text"], "created_at": None} for d in pool[:NDOCS - 1]]
    docs.sort(key=lambda d: h(c["cid"], "ord", d["chunk_id"]))     # shuffle so gold isn't always first
    items.append({"qid": "nut_" + c["cid"], "question": q, "answer": a, "aliases": [],
                  "docs": docs, "gold": [c["cid"]], "regime": "document"})
    if len(items) % 10 == 0:
        print("  built %d/%d (from %d attempts)" % (len(items), TARGET, used), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("WROTE %d items -> %s (attempts=%d)" % (len(items), OUT, used), flush=True)
if items:
    s = items[0]
    print("sample: Q=%r A=%r gold=%s ndocs=%d" % (s["question"][:80], s["answer"][:40], s["gold"], len(s["docs"])), flush=True)
