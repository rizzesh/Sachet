import argparse, glob, json, os, random, re
from collections import Counter, defaultdict
from generate import add_scores

random.seed(42)
DEFAULT_TEST = "lottery/prize,tech support,doctor's office,government survey"

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw/calls*.jsonl")
ap.add_argument("--out", default="data/processed")
ap.add_argument("--test", default=DEFAULT_TEST, help="comma-separated scenarios held out for test")
ap.add_argument("--val", default="", help="optional scenarios held out for validation")
a = ap.parse_args()

test_set = {s.strip() for s in a.test.split(",") if s.strip()}
val_set = {s.strip() for s in a.val.split(",") if s.strip()}


def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def combo(c):
    return (c["label"], c["scenario"], c.get("victim", "?"), c.get("language", "?"), c.get("pace", "?"))


files = sorted(glob.glob(a.raw))
print("reading:", files)
calls, seen, bad, old, dup = [], set(), 0, 0, 0
for p in files:
    with open(p, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                turns = c["turns"]
                callers = [t for t in turns if t["speaker"] == "caller"]
                ok = c.get("label") in ("scam", "normal") and c.get("scenario") and callers
            except (json.JSONDecodeError, KeyError, TypeError):
                ok = False
            if not ok:
                bad += 1
                print(f"skipped bad line {n} in {p}: {line[:60]}")
                continue
            if not any("signals" in t for t in callers):
                old += 1
                continue
            c["turns"] = add_scores(turns)
            key = norm(callers[0]["text"])[:60] + "|" + norm(turns[-1]["text"])[:40]
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            calls.append(c)
print(f"conversations: {len(calls)} | bad lines: {bad} | old-format skipped: {old} | duplicates removed: {dup}")

present = Counter(c["scenario"] for c in calls)
for s in sorted((test_set | val_set) - set(present)):
    print("WARNING: scenario not found in data:", s)

test = [c for c in calls if c["scenario"] in test_set]
val = [c for c in calls if c["scenario"] in val_set and c["scenario"] not in test_set]
rest = [c for c in calls if c["scenario"] not in test_set and c["scenario"] not in val_set]

if val_set:
    train = rest
else:
    groups = defaultdict(list)
    for c in rest:
        groups[combo(c)].append(c)
    keys = sorted(groups)
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
            "language": c.get("language", "?"), "victim": c.get("victim", "?"),
            "pace": c.get("pace", "?"), "turn_index": i, "n_turns": len(c["turns"]),
            "text_full": "\n".join(f"{x['speaker']}: {x['text']}" for x in part),
            "text_caller": "\n".join(f"caller: {x['text']}" for x in part if x["speaker"] == "caller"),
            "score": t["risk_score"],
        })
    return rows


os.makedirs(a.out, exist_ok=True)
for name, data in [("train", train), ("val", val), ("test", test)]:
    rows = [r for n, c in enumerate(data) for r in prefixes(c, f"{name}{n}")]
    with open(f"{a.out}/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lab = Counter(r["label"] for r in rows)
    avg = {k: round(sum(r["score"] for r in rows if r["label"] == k) / max(1, lab[k]), 1) for k in lab}
    calls_by_label = Counter(c["label"] for c in data)
    print(f"{name}: {len(data)} calls {dict(calls_by_label)}, {len(rows)} examples, avg score {avg}")
    print("   scenarios:", dict(Counter(c["scenario"] for c in data)))

if not any(c["label"] == "scam" for c in test) or not any(c["label"] == "normal" for c in test):
    print("WARNING: test set is missing scam or normal calls. Fix --test.")

hi = Counter(c["scenario"] for c in calls
             if c["label"] == "normal" and c["turns"][-1]["risk_score"] > 40)
print("normal calls ending above 40, by scenario:", dict(hi))