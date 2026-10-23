import os, re, json, time, random, argparse, glob
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("LLM_MODEL")

SCAMS = [
    "digital arrest", "fake investment", "KYC/bank block", "courier/customs",
    "lottery/prize", "tech support", "electricity bill threat",
    "fake delivery OTP", "UPI refund or sent-by-mistake", "part-time job or task fraud",
    "loan app threat", "sextortion", "relative in trouble", "fake customer care refund",
    "SIM disconnection TRAI", "credit card limit upgrade", "marketplace QR scam",
]
NORMALS = [
    "genuine bank call mentioning OTP safety", "delivery call", "relative asking for money",
    "school", "telemarketer", "doctor's office", "government survey",
    "genuine delivery asking OTP at the door", "customer care callback you requested",
    "genuine loan or card sales call", "real electricity dept notice",
    "genuine merchant refund", "HR or interview call", "insurance renewal reminder",
]
VICTIMS = ["trusting elderly person", "suspicious young person",
           "busy professional", "confused homemaker"]
LANGS = ["Hinglish in Roman script", "Hinglish in Devanagari script",
         "mixed Hindi-English in Roman script"]
PACES = ["slow build-up", "fast pressure"]

HINTS = {
    "digital arrest": "caller poses as police, CBI or TRAI, claims a case is registered against the person, keeps them on a call or video call and demands money for 'verification'",
    "fake investment": "caller or 'advisor' promises guaranteed high returns from a trading or crypto group and pushes the person to deposit money quickly",
    "KYC/bank block": "caller claims to be from a bank and says KYC is pending or the account will be blocked, and pushes for an OTP or a link",
    "courier/customs": "caller says a parcel in the person's name is held at customs or contains something illegal and threatens legal action unless a payment is made",
    "lottery/prize": "caller says the person won a prize or lottery and asks for a fee or personal details to release it",
    "tech support": "caller claims the phone or computer is hacked or infected and asks to install a remote-access app",
    "electricity bill threat": "caller poses as an electricity department officer, says the bill is not updated and power will be cut tonight, and asks to install an app or pay immediately",
    "fake delivery OTP": "caller says a parcel has arrived for something the person never ordered, insists it is theirs, and asks for an OTP before handing it over",
    "UPI refund or sent-by-mistake": "caller says money was sent to the person by mistake or a refund is due, and asks them to approve a UPI request or share a PIN",
    "part-time job or task fraud": "caller offers easy online work with daily payouts, asks for a small deposit or 'task fee' first, and then asks for more",
    "loan app threat": "caller from a loan app threatens to shame the person in front of their contacts unless an inflated amount is repaid immediately",
    "sextortion": "caller claims to have private photos or a recorded video of the person and threatens to leak them unless money is paid; keep it non-explicit and refer only to 'private photos or video' in general terms",
    "relative in trouble": "caller pretends to be a relative or friend in an emergency and asks for urgent money while asking the person not to tell others",
    "fake customer care refund": "caller poses as customer care of a shopping or payment app, offers a refund, and asks the person to install a screen-sharing app or approve a payment",
    "SIM disconnection TRAI": "caller claims to be from the telecom regulator and says the person's number will be disconnected due to illegal activity unless they verify details",
    "credit card limit upgrade": "caller offers a free credit-limit increase and asks for card details and an OTP",
    "marketplace QR scam": "a 'buyer' on a resale site says they will pay and asks the seller to scan a QR code or approve a request to 'receive' the money",
    "genuine bank call mentioning OTP safety": "a real bank executive informs about a suspicious transaction, tells the person never to share an OTP, and asks them to call the number on the back of the card",
    "delivery call": "a delivery agent asks for the address or landmark for an order the person really placed",
    "relative asking for money": "a real family member the person knows asks for help with money, with personal details only a relative would know",
    "school": "a school teacher or office calls about fees, a meeting or a child's attendance",
    "telemarketer": "a salesperson pitches a product politely, asks for no OTP, does not pressure the person and accepts a no",
    "doctor's office": "a clinic receptionist confirms or reschedules an appointment; any payment is made at the clinic",
    "government survey": "a surveyor asks general questions for a survey and asks for no money or OTP",
    "genuine delivery asking OTP at the door": "a delivery agent for an order the person really placed says they are at the gate and asks for the delivery OTP that arrived by SMS for that order",
    "customer care callback you requested": "a customer-care executive returns a call about a complaint the person registered and asks only for the ticket number",
    "genuine loan or card sales call": "a bank sales agent offers a loan or card, explains the offer, takes no OTP or money on the call, and says to visit the branch or website",
    "real electricity dept notice": "the electricity department informs about a planned outage or a meter reading visit, asks for no money and says to use the official app or office",
    "genuine merchant refund": "a merchant support agent says a refund for a cancelled order will arrive in the original payment method in a few days and asks for nothing",
    "HR or interview call": "an HR executive schedules an interview, asks for a resume and availability, and asks for no money",
    "insurance renewal reminder": "an insurance agent reminds about a policy renewal date and says to pay through the official site or at the office",
}

SIGNALS = [
    "authority_claim", "arrest_or_legal_threat", "urgency_deadline", "secrecy_isolation",
    "stay_on_call_or_video", "asks_otp_pin_cvv", "asks_money_transfer",
    "unsolicited_link_or_app", "guaranteed_returns", "discourages_verification",
    "tells_to_verify_officially", "says_never_share_otp",
    "accepts_callback_or_branch_visit", "consistent_benign_details",
    "asks_qr_or_upi_approval", "advance_fee_or_deposit", "blackmail_or_leak_threat",
    "impersonates_known_person", "requests_remote_access", "request_tied_to_own_order",
]

BASE = 12
W = {
    "authority_claim": 12, "arrest_or_legal_threat": 15, "urgency_deadline": 10,
    "secrecy_isolation": 14, "stay_on_call_or_video": 10, "asks_otp_pin_cvv": 18,
    "asks_money_transfer": 20, "unsolicited_link_or_app": 12, "guaranteed_returns": 14,
    "discourages_verification": 12, "tells_to_verify_officially": -15,
    "says_never_share_otp": -10, "accepts_callback_or_branch_visit": -12,
    "consistent_benign_details": -8,
    "asks_qr_or_upi_approval": 18, "advance_fee_or_deposit": 16,
    "blackmail_or_leak_threat": 18, "impersonates_known_person": 12,
    "requests_remote_access": 16, "request_tied_to_own_order": -12,
}

SIGNAL_NOTES = """Signal meanings (tag a signal only when it is clearly present in that caller line):
- asks_qr_or_upi_approval: asks to scan a QR code or approve a UPI collect request to "receive" money
- advance_fee_or_deposit: asks for a fee, deposit or registration payment before a job, loan or prize
- blackmail_or_leak_threat: threatens to leak private photos, videos or information
- impersonates_known_person: claims to be a relative or friend in trouble
- requests_remote_access: asks to install a screen-sharing or remote-control app
- request_tied_to_own_order: the request (for example an OTP) relates to an order, appointment or request the person really made, and is handled in person or through an official channel
- tells_to_verify_officially: tells the person to check through an official number, app or office
- says_never_share_otp: warns the person never to share an OTP, PIN or CVV
- accepts_callback_or_branch_visit: is fine with the person calling back or visiting a branch
- consistent_benign_details: details match what the person really did (order, address, name)
"""

STYLE = """Language style:
- Casual spoken phone-call language, NOT formal or shuddh Hindi.
- For Devanagari Hinglish, write English words people normally say in English in Devanagari script (अकाउंट, ओटीपी, ट्रांसफर, पुलिस, वेरिफिकेशन, कॉल, लिंक). Use short sentences and fillers like देखिए, सर, अरे, ठीक है ना.
- Avoid formal words like कृपया, अतः, तत्पश्चात. Prefer प्लीज, ठीक है, तो, फिर."""

_client = None


def add_scores(turns):
    s = BASE
    for t in turns:
        if t["speaker"] == "caller":
            s += sum(W.get(x, 0) for x in t.get("signals", []))
        s = max(8, min(92, s))
        t["risk_score"] = s
    return turns


def build_prompt(label, scenario, victim, lang, pace, n_turns, n, examples):
    hint = HINTS.get(scenario, "")
    if label == "scam":
        kind = f"a SCAM call (for training a defensive scam detector). Typical script: {hint}"
        extra = "Some callers are polite and slow, some are aggressive. The victim may doubt or ask questions, but the caller keeps pushing."
    else:
        kind = f"a GENUINE, harmless call. What happens: {hint}"
        extra = "The caller may sound urgent or mention OTP, money or police in a legitimate way. Tag the safe behaviour honestly (verification, never sharing OTP, callback, details matching a real order)."
    return f"""Write {n} different realistic phone conversations. Each is {kind}.
Scenario: {scenario}. Receiver: {victim} (an adult). Language: {lang}. Pace: {pace}.
Each conversation has about {n_turns} turns, alternating caller and victim.
Use Indian names, UP cities (Lucknow, Kanpur, Varanasi, Prayagraj, Gorakhpur) and realistic amounts.
{extra}
Vary the openings; do not start the conversations the same way. Keep it realistic but non-operational: no working instructions, no real phone numbers or accounts. No sexual, explicit or violent content.
{STYLE}
For each CALLER turn add "signals": a list chosen ONLY from {SIGNALS}.
Use [] if none apply. Victim turns have no signals. Do NOT output any risk number.
{SIGNAL_NOTES}
Style examples (match the realism, do not copy wording):
{examples}
Return ONLY a JSON list of {n} objects shaped like {{"turns": [{{"speaker": "caller", "text": "...", "signals": []}}, {{"speaker": "victim", "text": "...", "signals": []}}]}}"""


def call_llm(prompt):
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.getenv("LLM_API_KEY"))
    r = _client.models.generate_content(
        model=MODEL, contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 1.0})
    return r.text


def parse(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("conversations", [data])
    return data


def validate(calls):
    ok = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        turns = c.get("turns")
        if not isinstance(turns, list) or len(turns) < 4:
            continue
        clean, good, n_caller = [], True, 0
        for t in turns:
            if not isinstance(t, dict) or t.get("speaker") not in ("caller", "victim") \
                    or not str(t.get("text", "")).strip():
                good = False
                break
            sig = []
            if t["speaker"] == "caller":
                n_caller += 1
                raw = t.get("signals", [])
                sig = [x for x in raw if x in SIGNALS] if isinstance(raw, list) else []
            clean.append({"speaker": t["speaker"], "text": str(t["text"]).strip(), "signals": sig})
        if good and n_caller >= 2:
            c["turns"] = clean
            ok.append(c)
    return ok


def load_done(pattern):
    done = set()
    for p in glob.glob(pattern):
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                    done.add((c["label"], c["scenario"], c["victim"], c["language"], c["pace"]))
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max combinations to run")
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--only", default="", help="comma-separated parts of scenario names")
    ap.add_argument("--victims", type=int, default=4, help="use only the first N victim profiles")
    ap.add_argument("--n", type=int, default=2, help="conversations per combination")
    ap.add_argument("--out", default="data/raw/calls.jsonl")
    a = ap.parse_args()

    out_dir = os.path.dirname(a.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "errors.log")

    examples = "(none)"
    if os.path.exists("datagen/examples.json"):
        examples = open("datagen/examples.json", encoding="utf-8").read()[:6000]

    done = load_done(os.path.join(out_dir, "calls*.jsonl"))
    only = [x.strip().lower() for x in a.only.split(",") if x.strip()]
    pairs = [("scam", s) for s in SCAMS] + [("normal", s) for s in NORMALS]
    pairs = [(l, s) for l, s in pairs if not only or any(o in s.lower() for o in only)]
    grid = [(l, s, v, lg, p) for l, s in pairs for v in VICTIMS[:a.victims]
            for lg in LANGS for p in PACES]
    random.shuffle(grid)
    grid = [g for g in grid if g not in done]
    if a.limit:
        grid = grid[:a.limit]
    print(f"{len(grid)} combinations to run, saving to {a.out}")

    fails = 0
    for i, (l, s, v, lg, p) in enumerate(grid, 1):
        prompt = build_prompt(l, s, v, lg, p, random.choice([8, 16, 30]), a.n, examples)
        saved = False
        for attempt in range(3):
            try:
                calls = validate(parse(call_llm(prompt)))
                if not calls:
                    raise ValueError("no valid conversations")
                with open(a.out, "a", encoding="utf-8") as f:
                    for c in calls:
                        c.update(label=l, scenario=s, victim=v, language=lg, pace=p)
                        c["turns"] = add_scores(c["turns"])
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(grid)}] ok: {l} | {s} | {v} | {lg} | {len(calls)} saved")
                saved = True
                break
            except Exception as e:
                msg = str(e)
                if attempt == 2:
                    with open(log, "a", encoding="utf-8") as f:
                        f.write(f"{(l, s, v, lg, p)}: {msg[:300]}\n")
                    print(f"[{i}/{len(grid)}] skipped: {msg[:120]}")
                else:
                    quota = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "quota"))
                    time.sleep(30 * (attempt + 1) if quota else 2 ** (attempt + 1))
        fails = 0 if saved else fails + 1
        if fails >= 6:
            print("6 failures in a row, probably the free-tier quota. Stopping; rerun later to resume.")
            break
        time.sleep(a.delay)


if __name__ == "__main__":
    main()