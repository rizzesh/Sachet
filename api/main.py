import os, re, time, zlib, torch
import io
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/sachet_muril_v2")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")   # e.g. "your-hf-username/sachet-muril-v2"
HF_TOKEN      = os.getenv("HF_TOKEN", "")          # fine-grained token from https://huggingface.co/settings/tokens
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "false").lower() in ("true", "1", "yes")

API_KEY = os.getenv("SACHET_API_KEY", "")
LO, HI, MAXLEN = 8, 94, 512
MEDIUM, HIGH = 38, 62
EXPIRY = 30 * 60

# ── Dual-Mode Inference Engine (Local PyTorch vs Free HF Serverless API) ──────
tok = None
model = None
hf_client = None
stt = None

# If running in cloud / Render Free Tier or local model not present: use Serverless API (Zero RAM)
if HF_MODEL_REPO and (USE_HF_INFERENCE or not Path(MODEL_PATH).exists()):
    print(f"[Sachet] Free Serverless Inference Mode enabled for '{HF_MODEL_REPO}' (~45MB RAM)")
    try:
        from huggingface_hub import InferenceClient
        hf_client = InferenceClient(token=HF_TOKEN or None)
    except Exception as e:
        print(f"[Sachet] Warning initializing HF InferenceClient: {e}")
elif Path(MODEL_PATH).exists():
    # Local PyTorch mode (offline development on PC)
    print(f"[Sachet] Loading local PyTorch model from '{MODEL_PATH}'…")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        tok.truncation_side = "left"
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
    except Exception as e:
        print(f"[Sachet] Error loading local model: {e}")

def get_stt():
    """Lazy-load Whisper model only if audio endpoint is hit to conserve RAM on free tier."""
    global stt
    if stt is None:
        try:
            stt = WhisperModel(os.getenv("STT_MODEL", "tiny"), device="cpu", compute_type="int8")
        except Exception as e:
            print(f"[Sachet] Whisper init warning: {e}")
    return stt

NEG = re.compile(r"\b(mat|nahi|nahin|kabhi nahi|never|not|don'?t|na dein|nahi maangte|nahi maangta)\b|मत|नहीं|कभी नहीं", re.I)

SIG = {
  "authority": {
    "label": "Authority impersonation", "w": 10, "neg": False,
    "rx": (
      r"\b("
      r"police|pulis|inspector|sub[- ]?inspector|thanedar|hawaldar|constable|"
      r"dsp officer|sp officer|acp officer|dcp officer|police commissioner|"
      r"crime branch|special cell|cid officer|cbi officer|cbi|central bureau of investigation|"
      r"enforcement directorate|ed officer|ed raid|"
      r"intelligence bureau|ib officer|nia officer|ncrb|cyber cell|cyber police|cyber crime cell|"
      r"trai official|telecom regulatory|rbi officer|sebi officer|customs officer|customs department|"
      r"income tax officer|income tax department|it department officer|anti[- ]?narcotics|ncb officer|"
      r"supreme court|high court|district court|magistrate|judge sahab|adalat ka notice|"
      r"vigilance officer|nodal officer|fraud prevention officer"
      r")\b|पुलिस|सीबीआई|ईडी अधिकारी|कस्टम अधिकारी|अदालत|कोर्ट रूम|इंस्पेक्टर|साइबर सेल|आयकर अधिकारी|मजिस्ट्रेट|न्यायालय"
    ),
    "v": [
      "Caller khud ko Police/CBI/TRAI/Court officer bata raha hai — asli sarkari agencies phone par aise case banakar paise nahi maangti.",
      "Sarkari ya police adhikari hone ka dawa kiya ja raha hai; turant phone kaat kar official helpline par verify kijiye.",
      "Authority ka naam lekar dabav banana scam ka mukhya dhang hai."
    ]
  },
  "arrest": {
    "label": "Arrest / legal threat", "w": 14, "neg": False,
    "rx": (
      r"\b("
      r"arrest|arrested|giraftar|giraftari|hirasat|custody|remand|detention|detained|"
      r"warrant|arrest warrant|non[- ]?bailable|nbw|jail|prison|salakhon ke piche|"
      r"fir|case darj|case register|criminal case|chargesheet|challan|summons|court notice|"
      r"legal action|kanooni karwai|prosecution|action liya jayega|police aa rahi|police bhej|"
      r"money laundering|hawala|terror funding|drugs|charas|ganja|contraband|smuggling|"
      r"illegal parcel|illegal goods|child pornography|human trafficking|banned substance|"
      r"digital arrest|digital custody|virtual arrest|ghar baithe arrest|video arrest|video pe arrest|"
      r"saza hogi|hathkadi|andar kar dunga|fansa dunga"
      r")\b|अरेस्ट|गिरफ्तार|गिरफ्तारी|वारंट|जेल|हिरासत|मनी लॉन्ड्रिंग|हवाला|गैरकानूनी|केस दर्ज|चालान|समन|डिजिटल अरेस्ट|हथकड़ी|सज़ा|मुकदमा"
    ),
    "v": [
      "Arrest ya warrant ki dhamki di ja rahi hai — 'digital arrest' jaisi koi cheez Indian kanoon mein exist hi nahi karti.",
      "Giraftari ka darr dikhakar aapko jaldbazi aur dabav mein laane ki koshish ho rahi hai.",
      "Legal action ka darr: asli police phone par dhamki dekar settlement ya paise nahi maangti."
    ]
  },
  "secrecy": {
    "label": "Secrecy / isolation demand", "w": 8, "neg": False,
    "rx": (
      r"\b("
      r"kisi ko (mat|na) (bata|bol|kehna)|kisi ko mat batana|kisi ko na batao|mat batao|mat bolna|"
      r"kisi se (baat|share) mat|kisi tisre ko|third person ko|family ko (mat|na)|ghar walo ko (mat|na)|"
      r"apno ko (mat|na)|dosto ko (mat|na)|chup rehna|chup raho|shanti se|secretly|akele mein|akela|"
      r"confidential|classified|secret|top secret|private matter|investigation secret|"
      r"room band (kar|karke)|darwaza band|call record ho rahi|call tap ho rahi|sab sun rahe"
      r")\b|किसी को (मत|ना) बता|मत बताना|मत बोलो|गोपनीय|सीक्रेट|राज़|परिवार को मत|अकेले में|कमरा बंद"
    ),
    "v": [
      "Kisi ko na batane ko kaha ja raha hai — scammers victim ko parivaar aur apno se alag karte hain taaki koi madad na mil sake.",
      "Chup rehne aur 'confidential' hone ka dabav banaya ja raha hai; turant kisi trusted vyakti se baat kijiye.",
      "'Kisi se share mat karo' kehna scam ka clear sanket hai."
    ]
  },
  "stay_on_call": {
    "label": "Stay on call / video coercion", "w": 8, "neg": False,
    "rx": (
      r"\b("
      r"(call|phone|line) (mat|na) (kat|kaat|band|disconnect)|"
      r"line (par|pe) (bane|raho|rahiye)|stay on (call|line|phone)|don'?t (hang up|disconnect|cut)|"
      r"video call|camera on|webcam on|skype|face dikhao|screen (share|on)|"
      r"call kat gayi toh|phone kata toh|line disconnect hui toh"
      r")\b|फोन मत काटना|कॉल मत काटना|लाइन पर रहो|लाइन मत काटो|वीडियो कॉल|कैमरा ऑन|डिस्कनेक्ट मत करना"
    ),
    "v": [
      "Aapko lagatar phone ya video call par bane rehne ko bola ja raha hai — taaki aap kisi se salah na le sakein.",
      "Phone kaatne par aur zyada darr dikhaya ja raha hai taaki aap dabav mein rahein.",
      "Video call par lagatar nazar rakhna psychological coercion hai, turant call kaat dijiye."
    ]
  },
  "otp": {
    "label": "Asks OTP / PIN / credentials", "w": 18, "neg": True,
    "rx": (
      r"\b("
      r"otp|o\.?t\.?p|one[- ]time password|cvv|cvc|pin|m[- ]?pin|upi pin|atm pin|"
      r"password|passcode|security code|verification code|card number|16 digit|"
      r"card details|expiry date|valid through|debit card number|credit card number|"
      r"aadhaar number|aadhaar otp|pan card number|net banking password|login password"
      r")\b|ओटीपी|पिन|सीवीवी|पासवर्ड|पासकोड|कार्ड नंबर|आधार नंबर|वेरिफिकेशन कोड|सिक्योरिटी कोड"
    ),
    "v": [
      "OTP, PIN ya password maanga ja raha hai — ye KABHI kisi ko share mat kijiye, bank khata turant khali ho sakta hai.",
      "Koi bhi genuine bank ya officer phone par OTP nahi maangta. Turant call disconnect kijiye.",
      "Aapse confidential credentials maange gaye hain — savdhan rahiye."
    ]
  },
  "money": {
    "label": "Asks money transfer / payment", "w": 14, "neg": True,
    "rx": (
      r"\b("
      r"transfer|paisa (bhej|bhejo|daal|daalo|jama|transfer)|paise (bhej|bhejo|daal|daalo|jama|transfer)|"
      r"amount (bhej|daal|jama|transfer)|deposit|rtgs|neft|imps|upi|gpay|google pay|phonepe|paytm|"
      r"safe account|rbi account|government account|verification account|securing account|"
      r"court account|escrow|penalty (bharo|jama)|fine (bharo|jama)|bail amount|security deposit|"
      r"fd tod|fixed deposit tod|savings account se|kitna paisa hai|account balance"
      r")\b|ट्रांसफर|पैसे (भेज|भेजो|डाल|जमा)|सुरक्षित खाता|आरबीआई खाता|सरकारी खाता|पेमेंट करो|जुर्माना|पेनाल्टी"
    ),
    "v": [
      "Paisa kisi naye account ya safe account mein transfer karne ka dabav hai — sarkari agencies aisi koi cheez nahi karti.",
      "Kisi bhi verification ya settlement ke naam par paisa transfer mat kijiye.",
      "Paisa maangne ka dabav hai — ruk kar pehle official helpline se confirm kijiye."
    ]
  },
  "accidental_transfer": {
    "label": "Accidental transfer / refund claim", "w": 12, "neg": True,
    "rx": (
      r"\b("
      r"galti se.{0,45}(transfer|bhej|daal|chala gaya|chale gaye|ho gaya|ho gaye|aa gaye|aagaye|credit|paise)|"
      r"(transfer|bhej|daal|paise|amount|credit).{0,35}galti se|"
      r"galat (number|account|khata|upi|id).{0,35}(transfer|bhej|chala gaya|daal|chale gaye)|"
      r"accidentally.{0,30}(transferred|sent|paid|credited)|"
      r"(transferred|sent|paid|credited).{0,30}by mistake|"
      r"wrong (transfer|transaction|payment|upi transfer)|"
      r"mistakenly.{0,30}(transferred|sent|paid)|"
      r"mistake se.{0,35}(transfer|bhej|daal|ho gaya)|"
      r"wapas (bhej|transfer|kardo|kar dijiye|kar do|lauta)|"
      r"refund (kardo|kar dijiye|karo)|"
      r"extra (paise|amount|money).{0,30}(chala gaya|transfer ho)|"
      r"screenshot (dekho|dekhiye|bheja)|sms check (karo|kijiye)"
      r")\b|गलती से.{0,35}(पैसे|भेज|ट्रांसफर|चले गए|आ गए)|गलत नंबर पर|वापस (भेज|करो|दीजिए)|रिफंड करो"
    ),
    "v": [
      "Caller claim kar raha hai ki galti se paise transfer ho gaye aur wapas maang raha hai — ye aam 'Wrong Transfer / Fake SMS' scam hai.",
      "Pehle apna actual bank balance official netbanking app se khud check kijiye; kisi ke screenshot ya SMS par bharosa karke wapas transfer mat kijiye.",
      "Accidental Transfer Scam: Scammers fake transaction SMS dikhakar asali paise wapas mangte hain. Bank se bina confirm kiye ek rupaya bhi mat bhejiye."
    ]
  },
  "urgency": {
    "label": "Urgency / deadline pressure", "w": 6, "neg": False,
    "rx": (
      r"\b("
      r"turant|jaldi|abhi|abhi ke abhi|tatkal|fatafat|aaj raat|aaj hi|kal subah tak|"
      r"5 minute|10 minute|15 minute|20 minute|30 minute|aadha ghanta|1 ghanta|2 ghante|24 ghante|"
      r"immediately|right now|urgently|urgent|emergency|last chance|aakhri mauka|final warning|"
      r"deadline|time khatam|deri mat karo|der mat karo"
      r")\b|तुरंत|जल्दी|अभी के अभी|तत्काल|फटाफट|आज ही|आज रात|आखिरी मौका|समय नहीं है|डेडलाइन"
    ),
    "v": [
      "Jaldi karne ka bahut dabav hai — scammers sochne aur verify karne ka time nahi dena chahte.",
      "Time limit ya deadline bata kar panic create kiya ja raha hai; thoda ruk kar khud verify kijiye.",
      "Urgency ka darr manipulation ka aam tarika hai."
    ]
  },
  "link_app": {
    "label": "Remote access / suspicious app install", "w": 14, "neg": True,
    "rx": (
      r"\b("
      r"anydesk|teamviewer|quicksupport|rustdesk|airdroid|screen share|screenshare|remote access|remote control|"
      r"apk|download (karo|kijiye)|install (karo|kijiye)|link (par|pe) click|link bheja|link open|"
      r"play store se app|anayan app|whatsapp par bheja"
      r")\b|लिंक|डाउनलोड|इंस्टॉल|स्क्रीन शेयर|रिमोट|एनीडेस्क|टीमव्यूअर|क्विकसपोर्ट"
    ),
    "v": [
      "AnyDesk/TeamViewer ya screen-share app install karne ko kaha ja raha hai — isse scammer aapke phone ka control le sakta hai.",
      "Anjaan link ya APK file par click mat kijiye, ye malware ho sakta hai jo credentials chura leta hai.",
      "Remote access app install karwana account khali karne ka zariya hai."
    ]
  },
  "kyc_block": {
    "label": "KYC / SIM / account block threat", "w": 10, "neg": True,
    "rx": (
      r"\b("
      r"kyc|kyc update|kyc pending|kyc expire|kyc verification|"
      r"account (block|band|freeze|suspend|close)|card (block|band)|sim (block|band)|"
      r"service (band|block|suspend)|aadhaar link|pan link|"
      r"bijli|electricity|light bill|meter (kat|cut|disconnect|pending|update|reading)|"
      r"power (cut|grid|corporation)|bill pending|consumer id|connection (kat|cut|disconnect)"
      r")\b|केवाईसी|अकाउंट (ब्लॉक|बंद|फ्रीज)|सिम बंद|कार्ड ब्लॉक|बिजली (कट|बिल|बोर्ड)|कनेक्शन कट|मीटर"
    ),
    "v": [
      "KYC, SIM ya account block karne ki dhamki di ja rahi hai — bank ya service provider phone par aise block nahi karte.",
      "'Account band ho jayega' keh kar darana KYC scam ka classic script hai.",
      "Bijli/utility connection kaatne ki dhamki dekar turant payment mangna fraud hai."
    ]
  },
  "returns": {
    "label": "Guaranteed returns / task job scam", "w": 10, "neg": True,
    "rx": (
      r"\b("
      r"guarantee|guaranteed|assured|risk[- ]?free|munafa|double|triple|lakhpati|crorepati|"
      r"\d+ ?(percent|%) (return|profit|munafa)|daily income|daily profit|trading bot|crypto|bitcoin|"
      r"stock tip|telegram task|youtube (like|subscribe)|part time job|work from home job"
      r")\b|गारंटी|मुनाफा|रिटर्न|दोगुना|तिगुना|जोखिम मुक्त|क्रिप्टो|कमाई|पार्ट टाइम जॉब|यूट्यूब लाइक"
    ),
    "v": [
      "Guaranteed ya bahut zyada return ka lalach diya ja raha hai — investment mein koi bhi return guaranteed nahi hota.",
      "Telegram task ya YouTube like karke roz hazaron kamane ka offer fake job scam hai.",
      "Risk-free kamai ka daawa SEBI registration bina fraud hota hai."
    ]
  },
  "parcel": {
    "label": "Parcel / customs contraband claim", "w": 8, "neg": False,
    "rx": (
      r"\b("
      r"parcel|courier|consignment|customs|fedex|dhl|bluedart|speed post|"
      r"taiwan|dubai|thailand|narcotic|drugs|charas|ganja|contraband|illegal items|seized"
      r")\b|पार्सल|कूरियर|कस्टम|माल जब्त|नशीला|ड्रग्स|प्रतिबंधित"
    ),
    "v": [
      "Parcel mein 'illegal drugs/passport' nikalne ka darr dikhaya ja raha hai — ye FedEx/Customs parcel scam hai.",
      "Agar aapne koi international parcel nahi mangwaya toh bilkul mat dariye, ye fabricated script hai."
    ]
  },
  "lottery": {
    "label": "Prize / lottery fraud", "w": 10, "neg": False,
    "rx": (
      r"\b("
      r"lottery|kbc|kaun banega crorepati|prize|inaam|jackpot|lucky draw|lucky winner|"
      r"winner|bumper prize|25 lakh|cash prize|congratulations you won"
      r")\b|लॉटरी|इनाम|जैकपॉट|लकी ड्रा|विजेता|पुरस्कार|बधाई"
    ),
    "v": [
      "Inaam ya lottery jeetne ki baat ho rahi hai jo aapne kabhi bhari hi nahi — ye lottery scam hai.",
      "Prize lene ke badle processing fee ya tax ke naam par pehle paisa maangna fraud hota hai."
    ]
  },
  "tech_support": {
    "label": "Fake tech support claim", "w": 10, "neg": False,
    "rx": (
      r"\b("
      r"virus|malware|trojan|spyware|hacked|system compromised|laptop infected|"
      r"microsoft support|windows defender|tech support|technical department|firewall breach"
      r")\b|वायरस|हैक|मालवेयर|तकनीकी सहायता|सिस्टम खराब"
    ),
    "v": [
      "Device mein 'virus' ya 'hack' hone ka darr dikhakar remote control maanga ja raha hai.",
      "Microsoft ya Apple kabhi bhi users ko personally aisi calls nahi karte."
    ]
  },
  "victim_fear": {
    "label": "Victim fear / compliance state", "w": 8, "neg": False,
    "rx": (
      r"\b("
      r"dar lag raha|darr lag raha|dar gaya|ghabra gaya|ghabrahat|kya karun|kya karu|"
      r"please sir|bacha lijiye|bachao|meri koi galti nahi|maine kuch nahi kiya|main nirdosh|"
      r"police mat bhejo|jail mat bhejo|theek hai sir bhej raha|paise bhej deta|otp bata deta|"
      r"otp le lijiye|kar deta hu sir|ji sir bilkul|ha sir abhi karta"
      r")\b|डर लग रहा|घबराहट|परेशान|प्लीज सर|बचाओ|मेरी गलती नहीं|पैसे भेज देता|ओटीपी बता रहा|ठीक है सर"
    ),
    "v": [
      "Receiver ke darr aur compliance ke sanket hain — scammer isi ghabrahat ka fayda utha kar paise ya OTP mang raha hai. RUKIYE!",
      "Aap darr mein aakar bina soche kadam mat uthaiye — turant phone kaat kar kisi apne se baat kijiye."
    ]
  }
}

SAFE = {
  "verify": {
    "rx": r"khud (call|contact)|customer care|official (number|website|app|helpline)|branch (me|mein|jaiye|visit)|1930|toll[- ]?free|offishal|cybercrime\.gov|ब्रांच|कस्टमर केयर|ऑफिशियल|हेल्पलाइन"
  },
  "no_otp": {
    "rx": r"(otp|pin|cvv|password|ओटीपी|पिन).{0,40}(kabhi (nahi|mat)|never|don'?t share|nahi maang|nahi mangte|kisi ko mat batana|कभी नहीं|मत बताना|नहीं मांगते)"
  },
  "calm": {
    "rx": r"apni marzi|koi jaldi nahi|aaram se|soch lijiye|koi jabardasti nahi|apni convenience|branch aakar|अपनी मर्ज़ी|कोई जल्दी नहीं|सोच सकते हैं"
  }
}

SAFE_V = [
  "Caller ne official channel ya branch visit karne ki salah di — ye genuine vyavhaar hai.",
  "Caller ne OTP share na karne ki savdhani di — ye surakshit sanket hai."
]
NEUTRAL = [
  "Abhi tak koi gambhir khatre ka sanket nahi mila.",
  "Baatcheet filhaal normal lag rahi hai, par satark rahiye.",
  "Khatre ka koi clear pattern nahi mila."
]
GENERIC = [
  "Call ka dhang sandigdh lag raha hai — koi bhi confidential jankari ya paisa dene se pehle call kaat kar verify kijiye.",
  "Call mein dabav aur jaldbaazi ke lakshan hain; turant kisi parivaarjan se confirm kijiye."
]

COMBOS = [
  ({"authority", "arrest", "secrecy"}, [
    "DANGER: Classic Digital Arrest Scam! Police/CBI identity, giraftari ka darr, aur kamre mein chupchap rehne ka dabav — ye 100% scam hai, TURANT CALL KAATIYE!",
    "Digital Arrest Syndicate: Officer bankar jail ki dhamki aur kisi ko na batane ka aadesh. Indian law mein aisi koi prakriya nahi hoti."
  ]),
  ({"authority", "arrest", "money"}, [
    "CRITICAL THREAT: Digital Arrest Extortion — Police/CBI ka darr dikhakar aur jail ki dhamki dekar paise transfer karwaye ja rahe hain. TURANT CALL DISCONNECT KIJIYE!",
    "Digital Arrest Syndicate: Asli police phone par giraftari rokne ke badle paise nahi maangti."
  ]),
  ({"authority", "arrest"}, [
    "High Risk: Digital Arrest Pattern — CBI/Police hone ka dawa aur giraftari/FIR ka darr dikhaya ja raha hai.",
    "Authority + Legal Threat: Asli police phone par dara kar settlement ya case band karne ki baat nahi karti."
  ]),
  ({"accidental_transfer", "money"}, [
    "Accidental Transfer Scam: Caller dawa kar raha hai ki galti se paise transfer ho gaye aur wapas maang raha hai. Apna bank balance khud check kijiye, fake SMS ya screenshot ke jhaanse mein wapas paise mat bhejiye!",
    "Wrong Transaction Fraud: 'Galti se paise bhej diye' bolkar refund maangna aam fraud hai. Bina bank balance confirm kiye wapas transfer mat kijiye."
  ]),
  ({"accidental_transfer", "urgency"}, [
    "Urgent Refund Scam: Jaldbazi ya emergency ka bahana banakar 'galti se bheje paise' turant wapas maange ja rahe hain. Thoda ruk kar pehle bank statement verify kijiye."
  ]),
  ({"authority", "money"}, [
    "Authority + Payment Coercion: Adhikari ya agency ke naam par paise transfer karne ka dabav banaya ja raha hai. Asli agencies phone par payment nahi maangti.",
    "Suspicious Authority Payment: Kisi official ke naam par payment mangna fraud ka aam tarika hai."
  ]),
  ({"authority", "otp"}, [
    "CRITICAL ALERT: Officer bankar OTP/PIN maanga ja raha hai — ye bank khata khali karne ki saazish hai!"
  ]),
  ({"kyc_block", "otp"}, [
    "KYC Phishing Scam: Account/SIM block karne ka darr dikhakar OTP/PIN chheen-ne ka prayas."
  ]),
  ({"link_app", "tech_support"}, [
    "Remote Access Scam: Virus ka bahana banakar AnyDesk/TeamViewer se aapka phone/laptop hack kiya ja raha hai."
  ]),
  ({"returns", "money"}, [
    "Investment Fraud: Guaranteed return dikhakar pehle deposit karwaya ja raha hai — ye paisa kabhi wapas nahi milega."
  ]),
  ({"parcel", "arrest"}, [
    "FedEx/Customs Scam: Parcel mein drugs/narcotics hone ka jhooth bolkar police karwai ki dhamki di ja rahi hai."
  ]),
  ({"lottery", "money"}, [
    "Lottery Advance Fee Scam: Inaam/lucky draw ke badle pehle tax ya fee maangi ja rahi hai."
  ]),
  ({"victim_fear", "money"}, [
    "EMERGENCY: Scammer ke darr mein peedit paise transfer karne ki sthiti mein hai — TURANT PHONE DISCONNECT KIJIYE!"
  ]),
  ({"victim_fear", "otp"}, [
    "EMERGENCY: Coercion ke chalte OTP share kiya ja raha hai — Account turant compromise ho sakta hai!"
  ]),
  ({"kyc_block", "urgency"}, [
    "Utility/KYC Scam: Aaj raat connection ya khata band hone ki aafatafat machakar lootne ka prayas."
  ])
]

for d in SIG.values():
  d["rx"] = re.compile(d["rx"], re.I | re.DOTALL)
for d in SAFE.values():
  d["rx"] = re.compile(d["rx"], re.I | re.DOTALL)

app = FastAPI(title="Sachet API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
sessions = {}

# ── Serve the static frontend when deployed (client/ directory next to api/) ──
# NOTE: This block is intentionally placed BEFORE the @app routes so that
# the wildcard StaticFiles mount is registered LAST (after all /v1/* routes).
_CLIENT_DIR = Path(__file__).parent.parent / "client"
_SERVE_STATIC = _CLIENT_DIR.exists()

def _register_static_routes():
    if not _SERVE_STATIC:
        return

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(str(_CLIENT_DIR / "index.html"))

    @app.get("/{page}.html", include_in_schema=False)
    def serve_page(page: str):
        p = _CLIENT_DIR / f"{page}.html"
        if p.exists():
            return FileResponse(str(p))
        return FileResponse(str(_CLIENT_DIR / "index.html"))

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
  for k in [k for k, v in list(sessions.items()) if now - v["t"] > EXPIRY]:
    del sessions[k]

def detect(lines):
  all_text = "\n".join(lines).lower()
  caller_lines = [l[len("caller:"):].strip() for l in lines if l.startswith("caller:")]
  victim_lines = [l[len("victim:"):].strip() for l in lines if l.startswith("victim:")]
  n_caller = len(caller_lines)
  found, safe = {}, set()

  # 1. Scan caller utterances
  for i, text in enumerate(caller_lines):
    for sent in re.split(r"[.!?।\n]+", text):
      low = sent.lower().strip()
      if not low:
        continue
      neg = bool(NEG.search(low))
      for name, d in SIG.items():
        if name == "victim_fear":
          continue
        if d["rx"].search(low) and not (d["neg"] and neg):
          found[name] = i

  # 2. Scan victim utterances for fear / compliance signals
  for i, text in enumerate(victim_lines):
    low = text.lower().strip()
    if not low:
      continue
    if SIG["victim_fear"]["rx"].search(low):
      found["victim_fear"] = i

  # 3. Check entire dialogue context for any missed cross-sentence patterns (ignoring negatable signals)
  for name, d in SIG.items():
    if name not in found and name != "victim_fear" and not d["neg"]:
      if d["rx"].search(all_text):
        found[name] = n_caller - 1

  # 4. Check safe signals
  for name, d in SAFE.items():
    if d["rx"].search(all_text):
      safe.add(name)

  # 5. Suppress false positives when caller explicitly gives safety guidance
  if "no_otp" in safe and "otp" in found:
    if not ({"money", "link_app", "arrest"} & set(found.keys())):
      del found["otp"]

  return found, safe, n_caller

def calculate_heuristic_escalation(found, n_turns):
  if not found:
    return 0.0

  base_score = sum(SIG[k]["w"] for k in found)
  found_keys = set(found.keys())
  combo_boost = 0

  if {"authority", "arrest"} <= found_keys:
    combo_boost += 12
  if {"authority", "arrest", "secrecy"} <= found_keys:
    combo_boost += 8
  if {"accidental_transfer", "money"} <= found_keys:
    combo_boost += 12
  if {"kyc_block", "otp"} <= found_keys:
    combo_boost += 14
  if {"returns", "money"} <= found_keys:
    combo_boost += 12
  if ({"authority", "arrest"} & found_keys) and ({"money", "otp"} & found_keys):
    combo_boost += 16
  if {"link_app", "tech_support"} <= found_keys:
    combo_boost += 14
  if "urgency" in found_keys and ({"money", "otp", "arrest", "accidental_transfer"} & found_keys):
    combo_boost += 6
  if "victim_fear" in found_keys and len(found_keys) >= 2:
    combo_boost += 8

  # Progressive turn-aware ceiling to prevent hyper-jumping on turn 1 or 2
  raw = base_score + combo_boost
  if n_turns == 1:
    raw = min(raw, 32)
  elif n_turns == 2:
    raw = min(raw, 52)

  return float(raw)

def muril_inference(text):
    # 1. Local PyTorch model (if loaded on PC)
    if model is not None and tok is not None:
        try:
            enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
            with torch.inference_mode():
                p = model(**enc).logits.squeeze().clamp(0, 1).item()
            return p
        except Exception as e:
            print(f"[Sachet] Local inference error: {e}")

    # 2. Serverless Hugging Face Inference API (Free tier cloud - ~0 MB RAM)
    if hf_client is not None and HF_MODEL_REPO:
        try:
            res = hf_client.text_classification(text=text[:512], model=HF_MODEL_REPO)
            for item in res:
                lbl = getattr(item, "label", "") or (item.get("label", "") if isinstance(item, dict) else "")
                sc = getattr(item, "score", 0.0) or (item.get("score", 0.0) if isinstance(item, dict) else 0.0)
                if lbl.upper() in ("LABEL_1", "SCAM", "POSITIVE", "1"):
                    return float(sc)
            if res:
                first_score = getattr(res[0], "score", 0.5) if hasattr(res[0], "score") else res[0].get("score", 0.5)
                return float(first_score)
        except Exception as e:
            print(f"[Sachet] HF Serverless Inference warning: {e}")

    # 3. Graceful fallback
    return 0.5

def evaluate_risk(lines, sid):
  found, safe, n_turns = detect(lines)
  all_text = "\n".join(lines)

  muril_p = muril_inference(all_text)
  muril_val = muril_p * (HI - LO) + LO

  h_val = calculate_heuristic_escalation(found, n_turns)

  if h_val > 0:
    # Heuristic ratio scales moderately with signal diversity (from 0.35 to 0.65)
    ratio = min(0.35 + (len(found) - 1) * 0.10, 0.65)
    blended = (1.0 - ratio) * muril_val + ratio * h_val
    combined = max(blended, h_val * 0.88)
  else:
    combined = muril_val

  # Apply safe damping only if NO critical coercive actions have been detected
  critical_threats = {"arrest", "money", "otp", "link_app", "secrecy"}
  if safe and not (critical_threats & set(found.keys())):
    combined = max(LO, combined - 35)

  # Scale to bounds
  final_score = int(round(min(HI, max(LO, combined))))
  return final_score, found, safe

def level(s):
  return "high" if s >= HIGH else "medium" if s >= MEDIUM else "low"

def pick(options, sid, key):
  return options[zlib.crc32(f"{sid}:{key}".encode()) % len(options)]

def explain(sid, lines, lv, found, safe):
  if lv == "low" and not found:
    return pick(NEUTRAL, sid, "neutral"), []

  parts, covered = [], set()
  found_set = set(found.keys())

  # 1. Match combos in priority order (most specific first)
  for req, variants in sorted(COMBOS, key=lambda x: -len(x[0])):
    if req <= found_set:
      parts.append(pick(variants, sid, "combo" + "".join(sorted(req))))
      covered |= req
      break

  # 2. Add top category specific insights
  _, _, n_turns = detect(lines)
  ranked = sorted(found, key=lambda k: SIG[k]["w"] + (4 if found[k] >= n_turns - 2 else 0), reverse=True)
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

  # Re-evaluate on every turn to catch context
  sc, found, safe = evaluate_risk(s["lines"], c.session_id)

  # Gradual momentum without sudden spike
  if s["scores"]:
    prev_max = max(s["scores"])
    if prev_max >= HIGH and found and not safe:
      sc = max(sc, int(prev_max * 0.92))

  s["scores"].append(sc)
  lv = level(sc)
  reason, signals = explain(c.session_id, s["lines"], lv, found, safe)

  # Measured alert escalation:
  # Requires at least two consecutive turns in High Risk (>= 62),
  # OR turn >= 3 with a critical demand (money / otp / remote app) hitting severe risk (>= 75)
  sustained_high = len(s["scores"]) >= 2 and all(x >= HIGH for x in s["scores"][-2:])
  critical_demand = bool({"otp", "money", "link_app"} & set(found.keys())) and len(s["lines"]) >= 3 and sc >= 75
  confirmed = sustained_high or critical_demand

  return {
    "risk_score": sc,
    "level": lv,
    "reason": reason,
    "signals": signals,
    "alert": confirmed,
    "history": s["scores"]
  }

@app.post("/v1/session/end")
def end(e: End, x_api_key: str = Header(default="")):
  check_key(x_api_key)
  sessions.pop(e.session_id, None)
  return {"deleted": True}

@app.get("/health")
def health():
  return {"ok": True, "sessions": len(sessions)}

@app.post("/v1/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = "hi", x_api_key: str = Header(default="")):
  check_key(x_api_key)
  whisper_engine = get_stt()
  if whisper_engine is None:
    raise HTTPException(503, "Server Whisper STT not loaded. Use browser speech recognition.")
  content = await file.read()
  segs, _ = whisper_engine.transcribe(io.BytesIO(content), language=lang, vad_filter=True, beam_size=1)
  return {"text": " ".join(s.text.strip() for s in segs).strip()}

# ── Register static routes + file mount LAST (after all /v1/* routes) ─────────
_register_static_routes()
if _SERVE_STATIC:
    # Serve styles.css, api.html, github.html etc. at their relative paths.
    # Must be mounted AFTER all explicit routes so API routes always win.
    app.mount("/", StaticFiles(directory=str(_CLIENT_DIR), html=True), name="frontend")