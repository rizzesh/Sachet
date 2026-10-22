import os, json, time, random, argparse, re
from collections import defaultdict
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("LLM_API_KEY"))
MODEL = os.getenv("LLM_MODEL")
THRESH = 50

PROMPT = """You are a scam-call risk detector for Indian phone calls (Hindi/Hinglish).
Below is a call transcript so far. Rate how risky the call is RIGHT NOW as a score from 8 to 92.
Guide:
- Normal calls (delivery, school, doctor, relatives, genuine bank alerts) stay about 8-30.
- Risk rises with: claiming to be police/CBI/TRAI/bank head office, arrest or legal threats, urgent deadlines, "don't tell anyone", staying on a video call, asking for OTP/PIN, asking to transfer money to a "safe account", unknown links/apps, guaranteed returns, discouraging verification.
- Risk drops when the caller tells you to verify through official channels, says never to share OTP, or accepts a callback or branch visit.
- Judge the whole conversation and how it escalates, not single words.
Return ONLY JSON: {{"score": <integer 8-92>, "reason": "<one short line in simple Hinglish>"}}

Transcript:
{text}"""

def ask(text):
    r = client.models.generate_content(
        model=MODEL, contents=PROMPT.format(text=text),
        config={"response_mime_type": "application/json", "temperature": 0})
    m = re.search(r"\{.*\}", r.text, re.S)
    d = json.loads(m.group(0))
    return max(8, min(92, int(d["score"]))), d.get("reason", "")

ap = argparse.ArgumentParser()
ap.add_argument("--calls", type=int, default=20, help="number of test calls to evaluate")
ap.add_argument("--input", choices=["full", "caller"], default="full")
ap.add_argument("--delay", type=float, default=4.0)
a = ap.parse_args()

random.seed(1)
rows = [json.loads(l) for l in open("data/processed/test.jsonl", encoding="utf-8") if l.strip()]
calls = defaultdict(list)
for r in rows:
    calls[r["id"].rsplit("-", 1)[0]].append(r)
for k in calls:
    calls[k].sort(key=lambda r: r["turn_index"])

scam = [k for k in calls if calls[k][0]["label"] == "scam"]
norm = [k for k in calls if calls[k][0]["label"] == "normal"]
random.shuffle(scam); random.shuffle(norm)
chosen = scam[: a.calls // 2] + norm[: a.calls - a.calls // 2]
print(f"evaluating {len(chosen)} calls ({min(len(scam), a.calls // 2)} scam, {len(chosen) - min(len(scam), a.calls // 2)} normal)")

key = "text_full" if a.input == "full" else "text_caller"
preds = []
os.makedirs("eval", exist_ok=True)
with open(f"eval/preds_baseline_{a.input}.jsonl", "w", encoding="utf-8") as f:
    for ci, cid in enumerate(chosen, 1):
        for r in calls[cid]:
            try:
                s, why = ask(r[key])
            except Exception as e:
                print("error:", e)
                time.sleep(5)
                continue
            p = {"call": cid, "label": r["label"], "turn_index": r["turn_index"],
                 "n_turns": r["n_turns"], "true": r["score"], "pred": s, "reason": why}
            preds.append(p)
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            time.sleep(a.delay)
        print(f"call {ci}/{len(chosen)} done")

mae = sum(abs(p["true"] - p["pred"]) for p in preds) / len(preds)
by = defaultdict(list)
for p in preds:
    by[p["call"]].append(p)
sc = [v for v in by.values() if v[0]["label"] == "scam"]
nm = [v for v in by.values() if v[0]["label"] == "normal"]
alert = lambda v: any(x["pred"] >= THRESH for x in v)
recall = sum(alert(v) for v in sc) / max(1, len(sc))
false_alarm = sum(alert(v) for v in nm) / max(1, len(nm))
early = []
for v in sc:
    for x in v:
        if x["pred"] >= THRESH:
            early.append((x["turn_index"] + 1) / x["n_turns"])
            break
res = {"input": a.input, "threshold": THRESH, "calls": len(by), "rows": len(preds),
       "MAE_points": round(mae, 1), "scam_recall": round(recall, 3),
       "false_alarm_rate": round(false_alarm, 3),
       "avg_call_fraction_at_first_alert": round(sum(early) / len(early), 2) if early else None}
json.dump(res, open(f"eval/results_baseline_{a.input}.json", "w"), indent=2)
print(json.dumps(res, indent=2))