import csv, re

def nums(text):
    return {re.sub(r"\s+", "", n) for n in re.findall(r"\d[\d\s]*(?:[.,]\d+)?", text) if len(re.sub(r"\D","",n)) >= 2}

def refs(text):
    out = set()
    for m in re.finditer(r"п\.?\s*(\d+(?:\.\d+)+)", text): out.add(m.group(1))
    for m in re.finditer(r"ст\.?\s*(\d+(?:\.\d+)?)", text): out.add("ст." + m.group(1))
    for m in re.finditer(r"ред\.?\s*(20\d{2})", text): out.add("ред." + m.group(1))
    return out

def is_refusal(t):
    low = t.lower()
    return any(k in low for k in ("нет документов", "нет данных", "не могу ответить", "не отвечаю", "не хватает"))

gold = {r["id"]: r["answer"] for r in csv.DictReader(open("data/demo_output.csv", encoding="utf-8-sig"))}
ours = {r["id"]: r["answer"] for r in csv.DictReader(open("submission.csv", encoding="utf-8-sig"))}

stats = {"OK": 0, "PARTIAL": 0, "WRONG": 0, "MISSED": 0, "REFUSE_OK": 0, "REFUSE_BAD": 0}
rows = []
for qid in sorted(gold, key=int):
    g, o = gold[qid], ours.get(qid, "")
    g_ref, o_ref = is_refusal(g), is_refusal(o)
    if g_ref:
        verdict = "REFUSE_OK" if o_ref else "REFUSE_BAD"
    elif o_ref:
        verdict = "MISSED"
    else:
        n_hit = len(nums(g) & nums(o)); n_all = max(1, len(nums(g)))
        r_hit = len(refs(g) & refs(o)); r_all = max(1, len(refs(g)))
        if r_hit and n_hit: verdict = "OK"
        elif n_hit or r_hit: verdict = "PARTIAL"
        else: verdict = "WRONG"
    stats[verdict] += 1
    rows.append((qid, verdict, g[:70], o[:70]))

for qid, v, g, o in rows:
    print(f"{qid:>2} {v:11} | эталон: {g}")
    print(f"   {'':11} | наше:   {o}")
print()
print("итог:", stats)
total_good = stats["OK"] + stats["REFUSE_OK"]
print(f"полностью верных ответов: {total_good}/18")
