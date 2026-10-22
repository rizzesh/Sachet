import os, time, torch
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

HINTS = {
    "Police/CBI/TRAI jaisa authority claim": ["cbi", "police", "पुलिस", "trai", "ट्राई", "inspector", "इंस्पेक्टर", "cyber cell", "साइबर सेल", "court", "कोर्ट"],
    "Arrest ya legal dhamki": ["arrest", "अरेस्ट", "warrant", "वारंट", "गिरफ्तार"],
    "Kisi ko na batane ka dabav": ["mat bata", "kisi ko mat", "किसी को मत", "don't tell", "confidential"],
    "Video call par bane rehne ko kaha": ["video call", "वीडियो कॉल", "camera on", "कैमरा"],
    "OTP/PIN/CVV maanga": ["otp", "ओटीपी", "cvv", "pin "],
    "Paisa transfer karne ko kaha": ["transfer", "ट्रांसफर", "safe account", "सेफ अकाउंट"],
    "Jaldi karne ka dabav": ["turant", "तुरंत", "abhi", "अभी", "aaj raat", "आज रात", "15 minute"],
    "Pakka return ka vaada": ["guaranteed", "return", "रिटर्न", "profit"],
    "Link ya app install karne ko kaha": ["link", "लिंक", "app download"],
}

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

def hints(lines):
    caller = " ".join(l for l in lines if l.startswith("caller:")).lower()
    return [name for name, words in HINTS.items() if any(w in caller for w in words)]

def level(s):
    return "high" if s >= HIGH else "medium" if s >= MEDIUM else "low"

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
    found = hints(s["lines"])
    reason = "Abhi tak koi khaas khatre ka sanket nahi." if not found else "; ".join(found[:3])
    confirmed = len(s["scores"]) >= 2 and all(x >= HIGH for x in s["scores"][-2:])
    return {"risk_score": sc, "level": level(sc), "reason": reason,
            "signals": found, "alert": confirmed, "history": s["scores"]}

@app.post("/v1/session/end")
def end(e: End, x_api_key: str = Header(default="")):
    check_key(x_api_key)
    sessions.pop(e.session_id, None)
    return {"deleted": True}

@app.get("/health")
def health():
    return {"ok": True, "sessions": len(sessions)}