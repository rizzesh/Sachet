"""
Sachet - Synthetic Dialogue Generator (Phase 2)
Theme: Digital Safety & Cyber Fraud (VBYLD 2027)

Generates realistic Hindi/Hinglish phone call conversations across scam types
and hard-negative normal calls to train a defensive real-time risk detector.
Uses Google Gemini API via the google-genai SDK.

Risk scoring is deterministic: the LLM outputs signal tags per caller turn,
and score_turn() computes a 0-100 risk score using weighted signals.
"""

import os
import sys

# Configure UTF-8 encoding for Windows console to handle Hindi (Devanagari) properly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import time
import argparse
import itertools
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()

# Setup Google GenAI Client
from google import genai
from google.genai import types


# ---------------------------------------------------------
# Signal-Based Risk Scoring Constants & Function
# ---------------------------------------------------------

# Base risk that any unsolicited call starts at
BASE = 12

# Scam-indicative signal weights (positive = raises risk)
# Benign-indicative signal weights (negative = lowers risk)
W = {
    "authority_claim": 12,
    "arrest_or_legal_threat": 15,
    "urgency_deadline": 10,
    "secrecy_isolation": 14,
    "stay_on_call_or_video": 10,
    "asks_otp_pin_cvv": 18,
    "asks_money_transfer": 20,
    "unsolicited_link_or_app": 14,
    "guaranteed_returns": 12,
    "discourages_verification": 10,
    "tells_to_verify_officially": -15,
    "says_never_share_otp": -18,
    "accepts_callback_or_branch_visit": -12,
    "consistent_benign_details": -8,
}

# All valid signal names (for validation)
VALID_SIGNALS = set(W.keys())

# Decay factor: older signals lose influence
DECAY = 0.85


def score_turn(signals_so_far: List[List[str]]) -> int:
    """
    Deterministically computes a 0-100 risk score from all signals
    accumulated up to the current turn.

    signals_so_far: list of signal-lists, one per caller turn so far.
    Each inner list contains signal name strings for that turn.

    Uses exponential decay (older turns matter less), sums weighted
    signals, adds BASE, and clamps to [0, 100].
    """
    raw = BASE
    n = len(signals_so_far)
    for i, turn_signals in enumerate(signals_so_far):
        age = n - 1 - i  # 0 for most recent, higher for older
        weight_multiplier = DECAY ** age
        for sig in turn_signals:
            if sig in W:
                raw += W[sig] * weight_multiplier
    # Clamp to [5, 95] — ongoing calls always have uncertainty,
    # risk should never be absolute 0 or 100 mid-conversation
    return max(5, min(95, int(round(raw))))


# ---------------------------------------------------------
# Pydantic Schemas for JSON Validation & Constrained Decoding
# ---------------------------------------------------------

class Turn(BaseModel):
    speaker: str = Field(description="Must be 'caller' or 'victim'")
    text: str = Field(description="Utterance text in Hindi (Devanagari), Hinglish (Roman), or mixed Hindi-English")
    signals: List[str] = Field(
        default_factory=list,
        description="List of signal tags for caller turns (empty for victim turns). Valid signals: authority_claim, arrest_or_legal_threat, urgency_deadline, secrecy_isolation, stay_on_call_or_video, asks_otp_pin_cvv, asks_money_transfer, unsolicited_link_or_app, guaranteed_returns, discourages_verification, tells_to_verify_officially, says_never_share_otp, accepts_callback_or_branch_visit, consistent_benign_details"
    )

class Conversation(BaseModel):
    label: str = Field(description="'scam' or 'normal'")
    scenario: str
    language: str
    victim: str
    pace: str
    turns: List[Turn]

class LLMResponse(BaseModel):
    conversations: List[Conversation]


# ---------------------------------------------------------
# Output schema (what we write to JSONL — includes computed risk)
# ---------------------------------------------------------

class OutputTurn(BaseModel):
    speaker: str
    text: str
    signals: List[str]
    risk: int = Field(description="Computed 0-100 risk score")

class OutputConversation(BaseModel):
    label: str
    scenario: str
    language: str
    victim: str
    pace: str
    turns: List[OutputTurn]


# ---------------------------------------------------------
# LLM Caller Function
# ---------------------------------------------------------

def call_llm(prompt: str, model_name: Optional[str] = None) -> str:
    """
    Calls the Google Gemini LLM using the google-genai SDK.
    Reads LLM_API_KEY and LLM_MODEL from .env.
    Enforces structured JSON output matching LLMResponse schema.
    """
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY not found in environment or .env file.")

    if not model_name:
        model_name = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=LLMResponse,
        temperature=0.75,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    if not response.text:
        raise ValueError("Empty response received from LLM.")

    return response.text.strip()


# ---------------------------------------------------------
# Scenario Grid Definitions
# ---------------------------------------------------------

SCAM_TYPES = [
    "digital arrest",
    "KYC/bank block",
    "fake investment",
    "courier/customs",
    "electricity bill threat",
    "lottery/prize",
    "tech support",
]

NORMAL_TYPES = [
    "genuine bank call mentioning OTP safety",
    "delivery call",
    "relative asking for money",
    "telemarketer",
    "school",
    "doctor's office",
    "government survey",
]

VICTIM_PROFILES = [
    "trusting elderly person",
    "suspicious young person",
    "busy professional",
    "confused homemaker",
]

LANGUAGES = [
    "Hinglish in Roman script",
    "Hinglish in Devanagari script",
    "Mixed Hindi-English in Roman script",
]

PACES = [
    "slow build-up",
    "fast pressure",
]

TURN_LENGTHS = [8, 16, 30]


# ---------------------------------------------------------
# Domain Guidance (Authentic UP Cybercrime & Police Modus Operandi)
# ---------------------------------------------------------

DOMAIN_GUIDANCE = """
AUTHENTIC LANGUAGE & VOCABULARY GUIDELINES (REAL CASUAL SPOKEN PHONE CALLS):
1. For 'Hinglish in Devanagari script':
   - MUST be CASUAL spoken Hindi as spoken in everyday phone conversations.
   - ABSOLUTELY NO formal, bookish, or shuddh Hindi.
   - English words that people normally say in English MUST stay English but be written in Devanagari script:
     (अकाउंट, ओटीपी, ट्रांसफर, पुलिस, वेरिफिकेशन, कॉल, लिंक, कस्टमर, नंबर, ब्लॉक, ब्रांच, डिटेल्स, ऐप, पासवर्ड, मैसेज, पार्सल).
   - Use short sentences, natural fillers: "देखिए", "सर", "अरे", "सुनिए", "ठीक है ना", "अरे यार", "भैया", "हाँ जी".
   - Natural code-switching mix like real phone calls.
   - STRICTLY AVOID formal words: Do NOT use "कृपया", "आपका स्वागत है", "अतः", "तत्पश्चात", "पंजीकृत", "दूरसंचार विभाग", "अवैध वित्तीय गतिविधियों".
   - PREFER casual words: "प्लीज", "ठीक है", "तो", "फिर", "बंद हो जाएगा", "पैसे कट जाएंगे", "चेक कर लीजिए".
   - RIGHT EXAMPLE:
     "सर, आपका KYC अपडेट नहीं है, आज रात तक अकाउंट ब्लॉक हो जाएगा। ओटीपी बता दीजिए ना।"
   - WRONG EXAMPLE (Too formal - DO NOT USE):
     "कृपया ध्यान दें, आपके खाते का पुनर्सत्यापन अनिवार्य है अन्यथा खाता निष्क्रिय कर दिया जाएगा।"

2. For 'Hinglish in Roman script' & 'Mixed Hindi-English in Roman script':
   - Natural everyday Hinglish spoken in UP/India with natural code-switching.
   - Example: "Sir, aapka KYC pending hai, aaj raat tak account freeze ho jayega. Please OTP share kar dijiye na."
   - Avoid textbook phrases. Keep it colloquial, urgent or conversational, and authentic.

REALISTIC CONTEXT (UP Police & Cybercrime Helpline 1930 Advisories):
1. Digital Arrest Scams:
   - Callers claim to be from Telecom Regulatory Authority of India (TRAI), Alambagh Police Station (Lucknow), CBI, ED, ATS, or Barakhamba Police Station.
   - Fake officers like SI Sunil Kumar Mishra or Inspector Rathore.
   - Tell victims their mobile number/Aadhaar is linked to money laundering (e.g. Nawab Malik case or 17 complaints) and will be disconnected in 4 hours.
   - Demand victim switch to WhatsApp video call with fake police station background, demand 360-degree camera inspection to ensure victim is alone in the room.
   - Present fake Supreme Court or TRAI letters (often with typos), play walkie-talkie sounds in background.
   - Coerce victim to transfer all funds to 'RBI safe account' or 'government verification account' promising a clean-chit refund; pressure breaking FDs or taking loans.
   - REALITY: Police never do 'digital arrest' (not in BNSS/law), never take statements via video call, never send arrest warrants on WhatsApp, and never ask for funds transfer.

2. Hard Negative / Normal Calls:
   - Genuine bank calls: Actively caution the customer that bank staff NEVER ask for OTP, PIN, or CVV. Direct customer to call the number on the back of the debit card or visit their home branch.
   - Real deliveries: Delivery partner confirming landmark in UP localities (Gomti Nagar, Indira Nagar, Hazratganj, Civil Lines) without asking for OTPs or APK downloads.
   - Relatives: Calling naturally for urgent family assistance or loan request without fake legal threats.
   - Clinics / Schools: Legitimate appointment reminders or parent-teacher meeting notices.
"""


def build_prompt(
    scenario: str,
    label: str,
    victim: str,
    language: str,
    pace: str,
    target_turns: int,
) -> str:
    """
    Constructs the prompt for generating 2 distinct realistic conversations.
    Now asks for 'signals' per caller turn instead of a 'risk' number.
    """
    prompt = f"""You are a specialized training data generator for "Sachet", a defensive real-time scam-call risk detection service for Hindi/Hinglish calls.

Generate exactly 2 distinct, highly realistic phone conversations based on these specifications:
- Label: {label.upper()} ({label})
- Scenario: {scenario}
- Victim Profile: {victim}
- Language: {language}
- Escalation Pace: {pace}
- Target Length: approximately {target_turns} turns (total turns alternating between caller and victim).

{DOMAIN_GUIDANCE}

CRITICAL RULES:
1. Authentic Indian names (e.g., Ramesh, Sunita, Rajesh, Priya, Alok, Pooja, Vikram, Geeta, Veerendra) and UP cities (Lucknow, Kanpur, Varanasi, Prayagraj, Aligarh, Meerut, Bareilly, Agra).
2. TONE & VOCABULARY INSTRUCTION FOR '{language}':
   - If 'Hinglish in Devanagari script': Write CASUAL spoken Hindi, NEVER formal/shuddh Hindi. Common English words must stay in English transliterated to Devanagari (अकाउंट, ओटीपी, ट्रांसफर, पुलिस, वेरिफिकेशन, कॉल, लिंक, कस्टमर, नंबर, ब्लॉक, ऐप). Use short sentences and fillers ("देखिए", "सर", "अरे", "सुनिए", "ठीक है ना", "भैया"). Avoid formal words like "कृपया", "आपका स्वागत है", "अतः", "तत्पश्चात", "पंजीकृत". Prefer "प्लीज", "ठीक है", "तो", "फिर".
   - Target style example: "सर, आपका KYC अपडेट नहीं है, आज रात तक अकाउंट ब्लॉक हो जाएगा। ओटीपी बता दीजिए ना।"
   - If 'Hinglish in Roman script' or 'Mixed Hindi-English in Roman script': Use realistic natural conversational code-switching (e.g. "Sir, aapka KYC pending hai, aaj raat tak account freeze ho jayega. Please OTP share kar dijiye na.").
3. Defensive & non-operational: Realistic dialogues for detection training, no exploitative instructions.
4. Vary the openings and expressions. Do not repeat formulas across conversations.
5. Each turn must have:
   - "speaker": "caller" or "victim"
   - "text": The spoken line in {language}.
   - "signals": A list of signal tags for CALLER turns only. For VICTIM turns, use an empty list [].
     Valid signal names (use ONLY from this list, pick all that apply for that turn, or empty list if none):
       authority_claim, arrest_or_legal_threat, urgency_deadline, secrecy_isolation,
       stay_on_call_or_video, asks_otp_pin_cvv, asks_money_transfer, unsolicited_link_or_app,
       guaranteed_returns, discourages_verification,
       tells_to_verify_officially, says_never_share_otp, accepts_callback_or_branch_visit,
       consistent_benign_details.
   - Do NOT output any "risk" number. Only output "signals".
6. Output must follow the LLMResponse schema with exactly 2 conversations.
"""
    return prompt


# ---------------------------------------------------------
# Normalization & Validation Helper
# ---------------------------------------------------------

def sanitize_conversation(conv_data: Dict[str, Any], default_label: str, default_scenario: str, default_language: str = "Hinglish") -> Dict[str, Any]:
    """Sanitizes and bounds fields to ensure strict adherence to dataset standards."""
    clean_turns = []
    for t in conv_data.get("turns", []):
        raw_speaker = str(t.get("speaker", "caller")).lower().strip()
        speaker = "victim" if "victim" in raw_speaker or "user" in raw_speaker else "caller"
        text = str(t.get("text", "")).strip()

        # Extract and validate signals
        raw_signals = t.get("signals", [])
        if not isinstance(raw_signals, list):
            raw_signals = []
        # Filter to only valid signal names
        signals = [s for s in raw_signals if s in VALID_SIGNALS]
        # Victim turns should have no signals
        if speaker == "victim":
            signals = []

        if text:
            clean_turns.append({"speaker": speaker, "text": text, "signals": signals})

    return {
        "label": conv_data.get("label", default_label).lower(),
        "scenario": conv_data.get("scenario", default_scenario),
        "language": conv_data.get("language") or default_language,
        "victim": conv_data.get("victim", "general"),
        "pace": conv_data.get("pace", "standard"),
        "turns": clean_turns,
    }


def add_risk_scores(conv_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes a sanitized conversation dict (with 'signals' per turn)
    and computes a deterministic 'risk' score (0-100) for each turn
    using score_turn() based on accumulated caller signals.
    """
    caller_signals_so_far = []
    scored_turns = []

    for turn in conv_dict["turns"]:
        if turn["speaker"] == "caller" and turn["signals"]:
            caller_signals_so_far.append(turn["signals"])

        risk = score_turn(caller_signals_so_far)
        scored_turns.append({
            "speaker": turn["speaker"],
            "text": turn["text"],
            "signals": turn["signals"],
            "risk": risk,
        })

    return {
        "label": conv_dict["label"],
        "scenario": conv_dict["scenario"],
        "language": conv_dict["language"],
        "victim": conv_dict["victim"],
        "pace": conv_dict["pace"],
        "turns": scored_turns,
    }


# ---------------------------------------------------------
# Generation Engine with Resume, Validation, and Retries
# ---------------------------------------------------------

def load_existing_signatures(output_file: str) -> Set[str]:
    """
    Loads unique signatures of already generated calls for resume support.
    Signature format: scenario|victim|language|pace|label
    """
    signatures = set()
    if not os.path.exists(output_file):
        return signatures

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sig = f"{data.get('scenario')}|{data.get('victim')}|{data.get('language')}|{data.get('pace')}|{data.get('label')}"
                signatures.add(sig)
            except Exception:
                continue
    return signatures


def log_skip(skip_log_file: str, combo: Dict[str, Any], reason: str):
    """Logs skipped or failed combinations to disk."""
    os.makedirs(os.path.dirname(skip_log_file), exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "combo": combo,
        "reason": reason,
    }
    with open(skip_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_dataset(
    output_file: str = "data/raw/calls.jsonl",
    skip_log_file: str = "data/raw/skipped_log.jsonl",
    limit: Optional[int] = None,
    delay: float = 2.0,
    max_retries: int = 3,
):
    """
    Iterates through scenario grid, calls Gemini LLM, validates output,
    computes deterministic risk scores from signals, and persists to calls.jsonl.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    existing_sigs = load_existing_signatures(output_file)
    print(f"[*] Found {len(existing_sigs)} existing scenario signatures in {output_file}")

    # Build interleaved grid that alternates on EVERY step:
    # 1. Scenarios: alternates scam and normal calls across all 14 types
    # 2. Victims: cycles (elderly -> young -> professional -> homemaker)
    # 3. Languages: cycles (Hinglish Roman -> Hinglish Devanagari -> Mixed Roman)
    # 4. Paces: cycles (slow build-up -> fast pressure)
    # 5. Lengths: cycles (8 -> 16 -> 30)
    scams = [(s, "scam") for s in SCAM_TYPES]
    normals = [(n, "normal") for n in NORMAL_TYPES]
    interleaved_scenarios = [item for pair in zip(scams, normals) for item in pair]
    
    grid = []
    length_cycle = itertools.cycle(TURN_LENGTHS)
    
    for v_offset in range(len(VICTIM_PROFILES)):
        for l_offset in range(len(LANGUAGES)):
            for p_offset in range(len(PACES)):
                for s_idx, (scenario, label) in enumerate(interleaved_scenarios):
                    victim = VICTIM_PROFILES[(s_idx + v_offset) % len(VICTIM_PROFILES)]
                    lang = LANGUAGES[(s_idx + l_offset) % len(LANGUAGES)]
                    pace = PACES[(s_idx + p_offset) % len(PACES)]
                    target_len = next(length_cycle)
                    grid.append({
                        "scenario": scenario,
                        "label": label,
                        "victim": victim,
                        "language": lang,
                        "pace": pace,
                        "target_turns": target_len,
                    })

    print(f"[*] Total grid combinations: {len(grid)}")

    generated_count = 0
    combo_idx = 0

    for combo in grid:
        if limit is not None and generated_count >= limit:
            print(f"[*] Reached requested limit of {limit} conversations. Halting as instructed.")
            break

        sig = f"{combo['scenario']}|{combo['victim']}|{combo['language']}|{combo['pace']}|{combo['label']}"
        if sig in existing_sigs:
            # Already generated, resume next
            continue

        combo_idx += 1
        print(f"\n[{combo_idx}] Generating: [{combo['label'].upper()}] '{combo['scenario']}' | {combo['victim']} | {combo['language']} | {combo['pace']} (~{combo['target_turns']} turns)")

        prompt = build_prompt(
            scenario=combo["scenario"],
            label=combo["label"],
            victim=combo["victim"],
            language=combo["language"],
            pace=combo["pace"],
            target_turns=combo["target_turns"],
        )

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                raw_json = call_llm(prompt)
                parsed_data = json.loads(raw_json)

                # Validate and sanitize each conversation
                validated_convs = []
                for raw_conv in parsed_data.get("conversations", []):
                    clean_dict = sanitize_conversation(raw_conv, combo["label"], combo["scenario"], combo["language"])
                    # Validate against LLM schema (signals, no risk)
                    validated_model = Conversation.model_validate(clean_dict)
                    clean = validated_model.model_dump()
                    # Compute deterministic risk scores from signals
                    scored = add_risk_scores(clean)
                    validated_convs.append(scored)

                if not validated_convs:
                    raise ValueError("No conversations found in LLM output")

                # Append to calls.jsonl
                with open(output_file, "a", encoding="utf-8") as f_out:
                    for conv_dict in validated_convs:
                        f_out.write(json.dumps(conv_dict, ensure_ascii=False) + "\n")
                        generated_count += 1
                        # Show signal + risk summary for the conversation
                        total_signals = sum(len(t["signals"]) for t in conv_dict["turns"])
                        final_risk = conv_dict["turns"][-1]["risk"] if conv_dict["turns"] else 0
                        print(f"    -> Saved [{conv_dict['label']}] {len(conv_dict['turns'])} turns, {total_signals} signals, final_risk={final_risk}. (Total saved: {generated_count})")
                        if limit is not None and generated_count >= limit:
                            break

                existing_sigs.add(sig)
                success = True
                break

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"    [!] Attempt {attempt}/{max_retries} Schema error: {e}")
                if attempt == max_retries:
                    log_skip(skip_log_file, combo, f"Validation failure: {str(e)}")
            except Exception as e:
                print(f"    [!] Attempt {attempt}/{max_retries} API error: {e}")
                if attempt == max_retries:
                    log_skip(skip_log_file, combo, f"API error: {str(e)}")
            
            # Backoff before retry
            time.sleep(2.0 * attempt)

        # Rate limiting delay
        time.sleep(delay)

    print(f"\n[✓] Run completed. Generated {generated_count} new conversations. Total signatures: {len(existing_sigs)}.")


# ---------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Hindi/Hinglish scam and normal call dialogues.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of conversations to generate in this run.")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between LLM calls to respect rate limits.")
    parser.add_argument("--output", type=str, default="data/raw/calls.jsonl", help="Output JSONL path.")
    parser.add_argument("--skip-log", type=str, default="data/raw/skipped_log.jsonl", help="Path for logging failed combinations.")

    args = parser.parse_args()

    generate_dataset(
        output_file=args.output,
        skip_log_file=args.skip_log,
        limit=args.limit,
        delay=args.delay,
    )
    