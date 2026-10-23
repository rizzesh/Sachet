import os, re, time, zlib, torch
import io
from fastapi import UploadFile, File
from faster_whisper import WhisperModel

stt = WhisperModel(os.getenv("STT_MODEL", "small"), device="cpu", compute_type="int8")
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/sachet_muril")
API_KEY = os.getenv("SACHET_API_KEY", "")
LO, HI, MAXLEN = 8, 92, 512
MEDIUM, HIGH = 40, 65
EXPIRY = 30 * 60

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
tok.truncation_side = "left"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

NEG = re.compile(r"\b(mat|nahi|nahin|never|not|don'?t|na dein)\b|मत|नहीं", re.I)

SIG = {
 "authority": {"label": "Authority claim", "w": 3, "neg": False,
  "rx": r"\b(cbi|police|trai|inspector|cyber cell|court|ed officer|income tax|customs officer|rbi|head office|fraud department)\b|पुलिस|ट्राई|इंस्पेक्टर|साइबर सेल|कोर्ट|सीबीआई",
  "v": ["Caller khud ko police/CBI/TRAI jaisa officer bata raha hai — asli agencies phone par aisa nahi karti.",
        "Sarkari adhikari hone ka dava kiya ja raha hai; pehle khud official number par call karke confirm kijiye.",
        "Authority ka naam lekar dabav banaya ja raha hai, ye scam ka aam tarika hai."]},
 "arrest": {"label": "Arrest / legal threat", "w": 4, "neg": False,
  "rx": r"\b(arrest|warrant|jail|money laundering|legal action)\b|अरेस्ट|वारंट|गिरफ्तार|जेल|मनी लॉन्ड्रिंग",
  "v": ["Arrest/warrant ki dhamki di ja rahi hai — 'digital arrest' jaisi cheez kanoon mein hoti hi nahi.",
        "Giraftari ka darr dikhakar jaldi karwane ki koshish ho rahi hai.",
        "Legal dhamki di gayi hai; asli police phone par paisa nahi maangti."]},
 "secrecy": {"label": "Asked to keep secret", "w": 4, "neg": False,
  "rx": r"kisi ko mat|mat bata|kisi se baat mat|don'?t tell|confidential|secret|किसी को मत|मत बता|गोपनीय|कॉन्फिडेंशियल",
  "v": ["Kisi ko na batane ko kaha ja raha hai — scammers aksar victim ko apno se alag karte hain.",
        "Chup rehne ka dabav hai; kisi apne se zaroor poochhiye.",
        "'Confidential' bata kar aapko akela kiya ja raha hai."]},
 "stay_on_call": {"label": "Stay on call / video", "w": 3, "neg": False,
  "rx": r"(call|phone|line).{0,12}(mat kat|mat kaat|na kat)|line par bane|stay on|video call|camera on|वीडियो कॉल|कैमरा|फोन मत काट|कॉल मत काट",
  "v": ["Aapko call ya video par bane rehne ko kaha ja raha hai — taaki aap kisi se salah na le saken.",
        "Phone na kaatne ka dabav hai; asli officers aisa nahi karte.",
        "Video/call par lagatar nazar rakhi ja rahi hai — ye 'digital arrest' scam ka hissa hota hai."]},
 "otp": {"label": "Asks OTP / PIN / card details", "w": 5, "neg": True,
  "rx": r"\b(otp|cvv|pin|password|card number|card details|aadhaar number)\b|ओटीपी|सीवीवी|पिन|पासवर्ड|कार्ड नंबर",
  "v": ["OTP/PIN/CVV jaisi gopniya jaankari maangi ja rahi hai — ye kabhi kisi ko mat dijiye.",
        "Bank ya koi officer OTP phone par nahi maangta; ye scam ka bada sanket hai.",
        "Aapse OTP ya card details maangi gayi hain, turant ruk jaiye."]},
 "money": {"label": "Asks to transfer money", "w": 5, "neg": True,
  "rx": r"transfer|safe account|rbi account|government account|paisa (bhej|daal|jama)|paise (bhej|daal|jama)|deposit|ट्रांसफर|सेफ अकाउंट|पैसे (भेज|डाल|जमा)",
  "v": ["Paisa 'safe account' ya kisi naye account mein bhejne ko kaha ja raha hai — asli agency aisa nahi karti.",
        "Paise bhejne ka dabav hai; kuch bhi transfer karne se pehle ruk kar verify kijiye.",
        "'Verification' ke naam par paisa maanga ja raha hai, ye scam ka aakhri charan hota hai."]},
 "urgency": {"label": "Urgency pressure", "w": 2, "neg": False,
  "rx": r"turant|jaldi|abhi (ke abhi|hi)|aaj raat|12 baje|15 minute|10 minute|immediately|right now|last chance|only today|तुरंत|जल्दी|अभी के अभी|आज रात",
  "v": ["Jaldi karne ka dabav hai — scammers sochne ka time nahi dete.",
        "Time limit bata kar jaldbaazi karwayi ja rahi hai; thoda ruk kar verify kijiye.",
        "Turant action ka dabav dikh raha hai, isse risk badhta hai."]},
 "returns": {"label": "Guaranteed returns", "w": 4, "neg": True,
  "rx": r"guarantee|assured return|risk[- ]?free|munafa|double|\d+ ?(percent|%) (return|profit)|मुनाफा|गारंटी|रिटर्न",
  "v": ["Pakka/guaranteed return ka vaada kiya ja raha hai — investment mein koi return guaranteed nahi hota.",
        "Bahut zyada munafe ka laalach diya ja raha hai, ye fake investment scam ka sanket hai.",
        "Risk-free kamai ka dava ho raha hai; SEBI registration khud check kijiye."]},
 "link_app": {"label": "Link / app install", "w": 3, "neg": True,
  "rx": r"\blink\b|click|apk|anydesk|teamviewer|quicksupport|download|install|लिंक|डाउनलोड",
  "v": ["Link kholne ya app install karne ko kaha ja raha hai — anjaan link par click mat kijiye.",
        "Koi app install karwane ki koshish ho rahi hai; isse phone ka control ja sakta hai.",
        "Anjaan link ya app ka istemal karwaya ja raha hai, savdhan."]},
 "kyc_block": {"label": "KYC / account block threat", "w": 3, "neg": True,
  "rx": r"\bkyc\b|account (block|band|freeze|suspend)|card (block|band)|sim (block|band)|केवाईसी|अकाउंट (ब्लॉक|बंद)|ब्लॉक हो",
  "v": ["KYC/account block ka darr dikhaya ja raha hai; bank kabhi phone par aisi dhamki nahi deta.",
        "'Account band ho jayega' jaisi baat scam ka aam tarika hai; branch ya official app se check kijiye.",
        "Account/SIM block karne ki dhamki di gayi hai, pehle bank ke official number par call kijiye."]},
 "parcel": {"label": "Parcel / customs", "w": 1, "neg": False,
  "rx": r"parcel|courier|customs|fedex|dhl|narcotic|drugs|पार्सल|कूरियर|कस्टम",
  "v": ["Parcel/courier ke naam par baat ho rahi hai — agar aapne kuch mangaya nahi, to savdhan rahiye.",
        "Courier ya customs ka zikr hai; asli delivery mein paisa ya OTP phone par nahi maanga jata."]},
 "lottery": {"label": "Prize / lottery", "w": 3, "neg": False,
  "rx": r"lottery|prize|inaam|jackpot|lucky draw|winner|लॉटरी|इनाम|जैकपॉट",
  "v": ["Inaam/lottery jeetne ki baat ho rahi hai — jo aapne kabhi bhari hi nahi, wo jeetna mumkin nahi.",
        "Inaam ke badle paisa ya details maangna scam ka purana tarika hai."]},
 "electricity": {"label": "Electricity bill threat", "w": 3, "neg": False,
  "rx": r"bijli|electricity|light bill|connection (cut|kat)|बिजली|कनेक्शन (कट|कटेगा)",
  "v": ["Bijli bill aur connection kaatne ki dhamki di ja rahi hai — bijli vibhag aise phone par paisa nahi maangta.",
        "Bill ke naam par jaldi payment karwane ki koshish lag rahi hai; official app ya office se check kijiye."]},
 "tech_support": {"label": "Tech support claim", "w": 3, "neg": False,
  "rx": r"virus|hacked|microsoft|windows|technical support|वायरस|हैक",
  "v": ["Computer/phone mein 'virus' ya 'hack' bata kar darr dikhaya ja raha hai — ye tech-support scam ka tarika hai.",
        "Technical help ke naam par aapke device ka access maanga ja raha hai, savdhan."]},
}

SAFE = {
 "verify": {"rx": r"khud (call|contact)|customer care|official (number|website|app)|toll[- ]?free|branch (me|mein|ja|visit)|ब्रांच|कस्टमर केयर|ऑफिशियल"},
 "no_otp": {"rx": r"(otp|pin|cvv|ओटीपी).{0,30}(nahi maang|kabhi nahi|mat share|mat bata|never|नहीं मांग)"},
}
SAFE_V = ["Caller khud verify karne ya official channel use karne ko keh raha hai — ye achha sanket hai.",
          "Caller ne OTP share na karne ki salah di, ye asli bank/clinic jaisa vyavhaar hai."]
NEUTRAL = ["Abhi tak koi khaas khatre ka sanket nahi.", "Baatcheet normal lag rahi hai, par savdhan rahiye.",
           "Filhaal koi shak wali baat nahi mili."]
GENERIC = ["Call ka dhang shak wala lag raha hai, paisa ya OTP dene se pehle ruk jaiye.",
           "Is call mein kuch risk ke sanket hain; pehle kisi apne se ya official number se confirm kijiye."]
COMBOS = [
 ({"authority", "arrest"}, ["Ye 'digital arrest' scam jaisa lag raha hai — police/CBI ka naam, giraftari ka darr aur paisa maangna ek hi script ka hissa hain.",
                           "Digital arrest scam ka pattern: officer banke dara kar chup rehne aur paisa bhejne ko kaha jata hai."]),
 ({"kyc_block", "otp"}, ["KYC-update scam jaisa pattern: account block ka darr dikhakar OTP maanga ja raha hai."]),
 ({"returns", "money"}, ["Fake investment scam jaisa pattern: pakka munafa dikhakar paisa lagwane ki koshish."]),
 ({"parcel", "arrest"}, ["Courier/customs scam jaisa pattern: parcel mein 'gair-kanooni saaman' bata kar darr dikhaya ja raha hai."]),
 ({"lottery", "money"}, ["Lottery scam jaisa pattern: inaam ke badle pehle paisa maanga ja raha hai."]),
 ({"tech_support", "link_app"}, ["Tech-support scam jaisa pattern: virus ka darr dikhakar app install karwaya ja raha hai."]),
 ({"electricity", "urgency"}, ["Bijli bill scam jaisa pattern: connection kaatne ki dhamki dekar jaldi payment ki baat."]),
]

for d in SIG.values():
    d["rx"] = re.compile(d["rx"], re.I)
for d in SAFE.values():
    d["rx"] = re.compile(d["rx"], re.I)

app = FastAPI(title="Sachet API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
sessions = {}

class Chunk(BaseModel):
    session_id: str
    text: str
    speaker: str = "caller"

class End(BaseModel):
    session_id: str

def check_key(k):
    if API_KEY and k != API_KEY:
        raise HTTPException(401, "bad api key")

def purge():
    now = time.time()
    for k in [k for k, v in sessions.items() if now - v["t"] > EXPIRY]:
        del sessions[k]

def score(text):
    enc = tok(text, truncation=True, max_length=MAXLEN, return_tensors="pt")
    with torch.no_grad():
        p = model(**enc).logits.squeeze().clamp(0, 1).item()
    return round(p * (HI - LO) + LO)

def level(s):
    return "high" if s >= HIGH else "medium" if s >= MEDIUM else "low"

def pick(options, sid, key):
    return options[zlib.crc32(f"{sid}:{key}".encode()) % len(options)]

def detect(lines):
    caller = [l[len("caller:"):].strip() for l in lines if l.startswith("caller:")]
    found, safe = {}, set()
    for i, text in enumerate(caller):
        for sent in re.split(r"[.!?।\n]+", text):
            low = sent.lower().strip()
            if not low:
                continue
            neg = bool(NEG.search(low))
            for name, d in SIG.items():
                if d["rx"].search(low) and not (d["neg"] and neg):
                    found[name] = i
            for name, d in SAFE.items():
                if d["rx"].search(low):
                    safe.add(name)
    return found, safe, len(caller)

def explain(sid, lines, lv):
    if lv == "low":
        return pick(NEUTRAL, sid, "neutral"), []
    found, safe, n = detect(lines)
    parts, covered = [], set()
    for req, variants in COMBOS:
        if req <= set(found):
            parts.append(pick(variants, sid, "combo" + "".join(sorted(req))))
            covered |= req
            break
    ranked = sorted(found, key=lambda k: SIG[k]["w"] + (3 if found[k] >= n - 2 else 0), reverse=True)
    for name in ranked:
        if len(parts) >= 2:
            break
        if name not in covered:
            parts.append(pick(SIG[name]["v"], sid, name))
    if not parts:
        parts.append(pick(GENERIC, sid, "generic"))
    if lv != "high" and safe:
        parts.append(pick(SAFE_V, sid, "safe"))
    return " ".join(parts), [SIG[k]["label"] for k in found]

@app.post("/v1/analyze")
def analyze(c: Chunk, x_api_key: str = Header(default="")):
    check_key(x_api_key)
    purge()
    s = sessions.setdefault(c.session_id, {"lines": [], "scores": [], "t": time.time()})
    s["lines"].append(f"{c.speaker}: {c.text}")
    s["t"] = time.time()
    if c.speaker == "caller" or not s["scores"]:
        s["scores"].append(score("\n".join(s["lines"])))
    sc = s["scores"][-1]
    lv = level(sc)
    reason, signals = explain(c.session_id, s["lines"], lv)
    confirmed = len(s["scores"]) >= 2 and all(x >= HIGH for x in s["scores"][-2:])
    return {"risk_score": sc, "level": lv, "reason": reason, "signals": signals,
            "alert": confirmed, "history": s["scores"]}

@app.post("/v1/session/end")
def end(e: End, x_api_key: str = Header(default="")):
    check_key(x_api_key)
    sessions.pop(e.session_id, None)
    return {"deleted": True}

@app.get("/health")
def health():
    return {"ok": True, "sessions": len(sessions)}
    
@app.post("/v1/transcribe")
def transcribe(file: UploadFile = File(...), lang: str = "hi", x_api_key: str = Header(default="")):
    check_key(x_api_key)
    segs, _ = stt.transcribe(io.BytesIO(file.file.read()), language=lang, vad_filter=True, beam_size=1)
    return {"text": " ".join(s.text.strip() for s in segs).strip()}