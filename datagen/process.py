import json, os, random, re
from collections import Counter, defaultdict

BASE = 12
W = {"authority_claim": 12, "arrest_or_legal_threat": 15, "urgency_deadline": 10,
     "secrecy_isolation": 14, "stay_on_call_or_video": 10, "asks_otp_pin_cvv": 18,
     "asks_money_transfer": 20, "unsolicited_link_or_app": 12, "guaranteed_returns": 14,
     "discourages_verification": 12, "tells_to_verify_officially": -15,
     "says_never_share_otp": -10, "accepts_callback_or_branch_visit": -12,
     "consistent_benign_details": -8}

def add_scores(turns):
    s = BASE
    for t in turns:
        if t["speaker"] == "caller":
            s += sum(W.get(x, 0) for x in t.get("signals", []))
        s = max(8, min(92, s))
        t["risk_score"] = s
    return turns

random.seed(42)
RAW = "data/raw/calls.jsonl"
OUT = "data/processed"
TEST_SCENARIOS = {"courier/customs", "lottery/prize", "delivery call", "doctor's office"}
def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()

import glob
files = sorted(glob.glob("data/raw/calls*.jsonl"))
print("reading:", files)
calls, seen, bad, old = [], set(), 0, 0
for p in files:
    for n, line in enumerate(open(p, encoding="utf-8-sig"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            caller = [t["text"] for t in c["turns"] if t["speaker"] == "caller"]
            if not any("signals" in t for t in c["turns"] if t["speaker"] == "caller"):
                old += 1
                continue
            c["turns"] = add_scores(c["turns"])
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
            print(f"skipped bad line {n} in {p}: {line[:60]}")
            continue
        if not caller:
            continue
        key = norm(caller[0])[:60] + "|" + norm(c["turns"][-1]["text"])[:40]
        if key in seen:
            continue
        seen.add(key)
        calls.append(c)
print("conversations after dedupe:", len(calls), "| bad lines:", bad, "| old-format skipped:", old)

def combo(c):
    return (c["label"], c["scenario"], c["victim"], c["language"], c["pace"])

test = [c for c in calls if c["scenario"] in TEST_SCENARIOS]
rest = [c for c in calls if c["scenario"] not in TEST_SCENARIOS]
groups = defaultdict(list)
for c in rest:
    groups[combo(c)].append(c)
keys = list(groups)
random.shuffle(keys)
cut = int(len(keys) * 0.85)
train = [c for k in keys[:cut] for c in groups[k]]
val = [c for k in keys[cut:] for c in groups[k]]

def prefixes(c, cid):
    rows = []
    for i, t in enumerate(c["turns"]):
        if t["speaker"] != "caller":
            continue
        part = c["turns"][: i + 1]
        rows.append({
            "id": f"{cid}-{i}", "label": c["label"], "scenario": c["scenario"],
            "language": c["language"], "turn_index": i, "n_turns": len(c["turns"]),
            "text_full": "\n".join(f"{x['speaker']}: {x['text']}" for x in part),
            "text_caller": "\n".join(f"caller: {x['text']}" for x in part if x["speaker"] == "caller"),
            "score": t["risk_score"],
        })
    return rows

os.makedirs(OUT, exist_ok=True)
for name, data in [("train", train), ("val", val), ("test", test)]:
    rows = [r for n, c in enumerate(data) for r in prefixes(c, f"{name}{n}")]
    with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lab = Counter(r["label"] for r in rows)
    sc = {k: round(sum(r["score"] for r in rows if r["label"] == k) / max(1, lab[k]), 1) for k in lab}
    print(f"{name}: {len(data)} calls, {len(rows)} examples, labels {dict(lab)}, avg score {sc}")

normal_final = [c["turns"][-1]["risk_score"] for c in calls if c["label"] == "normal"]
if normal_final:
    high = sum(1 for s in normal_final if s > 40)
    print(f"normal calls ending above 40: {high}/{len(normal_final)}")