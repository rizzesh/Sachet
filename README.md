# Sachet (सचेत)

**Real-Time Scam-Call Risk Detector for Hindi and Hinglish Conversations**  
*Defensive Use Only | "Warn, Never Block"*

---

## Overview

Sachet is an intelligent, privacy-first conversational risk detection system designed to safeguard citizens against evolving voice scams conducted in Hindi and Hinglish (such as Digital Arrest, fake police/CBI extortion, electricity disconnection threats, and accidental transfer scams).

Rather than relying on static phone number blacklists or superficial keyword matching, Sachet analyzes the progressive psychological escalation of the conversation across multiple turns. It tracks authority impersonation, urgency induction, forced isolation, and financial demands, providing:

- **Dynamic Risk Score**: Continuous numeric rating from 8 to 94.
- **Categorical Risk Level**: `Normal`, `Caution`, or `High Risk`.
- **Explainable Hindi/Hinglish Warnings**: Context-specific reasoning explaining the exact deception tactic being attempted.
- **Immediate Advisory Actions**: Direct pointers to official reporting channels (National Cyber Crime Helpline 1930 and cybercrime.gov.in).

---

## Key Scenarios Covered

Sachet is trained and validated across 31 distinct conversational scenarios:

### Scam Scenarios (17)
1. **Digital Arrest**: Posing as Police, CBI, ED, or Court officials claiming an arrest warrant; demanding the victim remain on video call.
2. **Fake Investment Fraud**: Promising guaranteed high returns through illicit trading, crypto, or Telegram groups.
3. **KYC / Account Block Threat**: Claiming a bank account, SIM card, or PAN will be frozen unless verified immediately.
4. **Courier / Customs Parcel Extortion**: Falsely alleging illegal contraband or drugs were seized in a parcel bearing the victim's name.
5. **Electricity Disconnection**: Warning that household power will be cut tonight unless immediate payment or app download occurs.
6. **UPI Refund / Sent-by-Mistake**: Falsely claiming money was mistakenly transferred to the victim and requesting a refund via UPI collect.
7. **Lottery / Prize Fraud**: Promising substantial cash prizes or lucky draws conditioned on paying advance processing fees or taxes.
8. **Tech Support / Malware Alert**: Impersonating Microsoft or device security teams to coerce victims into installing remote access apps.
9. **Fake Delivery OTP**: Delivery agent demanding cancellation or verification OTPs for unrequested shipments.
10. **Part-Time Task Fraud**: Offering daily earnings for video likes or reviews, followed by escalating prepaid deposit demands.
11. **Loan App Extortion**: Harassment and blackmail over inflated micro-loans using contact lists.
12. **Sextortion**: Threatening to leak private photos or videos unless ransom is paid.
13. **Relative in Emergency**: Posing as a family member involved in an accident or police custody pleading for urgent financial help.
14. **Fake Merchant Refund**: Impersonating shopping or payment app support offering a refund via screen-sharing apps.
15. **TRAI SIM Termination**: Threatening telecom disconnection within hours over alleged illegal activity.
16. **Credit Card Upgrade**: Soliciting 16-digit card numbers, CVVs, and netbanking passwords under the guise of limit enhancements.
17. **Marketplace QR Scam**: Posing as a prospective buyer on resale platforms and asking the seller to scan a QR code to "receive" money.

### Genuine Control Scenarios (14)
1. **Legitimate Bank Call with Safety Warning**: Informing about suspicious activity while explicitly warning never to share OTPs or PINs.
2. **Standard E-Commerce Delivery**: Confirming address landmarks for genuine orders.
3. **Doorstep Delivery with Code**: In-person handover requesting delivery OTP tied to an active order.
4. **Authentic Family Assistance**: Real family members discussing personal finances without secrecy demands.
5. **School / College Administration**: Routine administrative notices regarding meetings or fee schedules.
6. **Medical Clinic Confirmation**: Confirming or rescheduling doctor appointments with payment at reception.
7. **Polite Telemarketing**: Unpressured product pitches that respect negative responses and do not request credentials.
8. **Government Survey**: Demographic or census questions without asking for personal banking data.
9. **Requested Customer Support Callback**: Follow-ups referencing specific ticket numbers created by the user.
10. **Standard Bank Loan Promotion**: Informing about pre-approved offers and directing customers to visit official branches.
11. **Official Utility Notice**: Planned maintenance announcements advising the use of official state portal apps.
12. **Merchant Order Refund**: Informative updates regarding automated refunds returning to original payment sources.
13. **HR Interview Scheduling**: Professional recruitment calls requesting resumes and interview time slots.
14. **Insurance Policy Renewal**: Routine expiration reminders directing payments to official insurer portals.

---

## Technical Architecture

```text
               +-------------------------------------------+
               |  Speech Audio (Web Speech API / Whisper)  |
               +-------------------------------------------+
                                     |
                                     v
                        +-------------------------+
                        |  Turn Transcript Text   |
                        +-------------------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
    +--------------------------+           +--------------------------+
    |  Fine-Tuned MuRIL Model  |           |  Heuristic Signal Engine |
    |  (Hinglish Transformer)  |           |  - Authority & Arrest    |
    |  Sequence Classification |           |  - Urgency & Secrecy     |
    +--------------------------+           |  - Financial & OTP Demands|
                 |                         |  - Negation Filtering    |
                 |                         +--------------------------+
                 |                                       |
                 +-------------------+-------------------+
                                     |
                                     v
               +-------------------------------------------+
               |         Hybrid Compound Scoring           |
               | S_final = (1 - a) * S_MuRIL + a * S_heur |
               |     Turn-aware damping & confirmations    |
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |  FastAPI Output: Score, Level & Reason    |
               +-------------------------------------------+
```

### 1. Indic Language Model: Fine-Tuned MuRIL
We fine-tuned Google's **MuRIL** (`google/muril-base-cased`) for multi-turn sequence classification over Romanized and Devanagari Hindi. MuRIL was chosen over standard multilingual models due to its pre-training on 17 Indian languages and its superior handling of code-mixed phonetics.

### 2. Heuristic Signal & Escalation Engine
To ensure explainability and prevent abrupt scoring spikes on single ambiguous utterances, Sachet combines neural classification with deterministic rule graphs:
- **Negation Masking**: Suppresses false positives when safe phrases occur (e.g., *"bank kabhi OTP nahi maangta"*).
- **Compound Escalation**: Boosts risk non-linearly when multiple distinct threat vectors converge (e.g., *Authority + Arrest Threat + Demand for Secrecy*).
- **Turn-Aware Progression**: Imposes calibrated ceilings on early turns to prevent premature alerts before sufficient conversational context is established.

### 3. Dual-Mode Deployment Architecture
- **Local Development**: Runs local PyTorch inference using local model checkpoints.
- **Cloud Free-Tier Mode**: Connects to Hugging Face Serverless Inference over HTTP, reducing server RAM requirements from **1.3 GB down to ~45 MB**. This ensures deployment stability on free-tier containers without memory limit crashes.

---

## Model Evaluation Results

Evaluation conducted on a held-out test suite of 92 multi-turn calls comparing our fine-tuned MuRIL model against the multilingual baseline:

| Metric | XLM-RoBERTa Baseline | Fine-Tuned MuRIL (Sachet) |
|---|---|---|
| **Scam Recall (Test)** | 86.4% | **95.5%** |
| **Scam Recall (Validation)** | 90.0% | **96.6%** |
| **Mean Absolute Error (MAE)** | 14.1 points | **7.4 – 10.2 points** |
| **First Alert Call Fraction** | 0.49 (late) | **0.32 – 0.46 (early)** |
| **Inference Latency** | ~320 ms | **< 150 ms** |

Sachet consistently triggers alerts within the first **32% to 46%** of the call duration, warning users before they reach the payment or credential disclosure stage.

---

## Privacy by Design

Sachet adheres to strict privacy safeguards:
- **No Audio Recording**: Audio is processed ephemerally on the client or via local streaming transcription; no raw audio files are stored.
- **In-Memory Session State**: Active conversation history resides strictly in volatile RAM.
- **Automatic Expiry**: Sessions are deleted after 30 minutes of inactivity.
- **Explicit Deletion**: Calling `/v1/session/end` immediately drops the session from memory.
- **Local Evidence Export**: Transcripts and scores can be saved locally by the user as plain text for evidence reporting. Nothing is retained on the server.

---

## Repository Structure

```text
Sachet/
├── api/
│   └── main.py              # FastAPI server, heuristic scoring, and static file mount
├── client/
│   ├── index.html           # Interactive web dashboard (gauge, live transcript, demo calls)
│   ├── styles.css           # Responsive styling and dark theme
│   ├── api.html             # REST API endpoint documentation
│   └── how-it-works.html    # Architectural details and threat vector explanations
├── data/
│   ├── processed/           # Processed and split training/validation datasets
│   └── splits/              # Leakage-free train/test splits
├── datagen/
│   └── generate.py          # Synthetic conversational dialogue generator
├── eval/
│   ├── baseline.py          # Baseline and comparative evaluation scripts
│   ├── results_fine-tuned_muril.jsonl
│   └── results_xlm-r.jsonl
├── models/                  # Local model weights directory (ignored from Git)
├── Dockerfile               # Container definition for containerized deployments
├── requirements.txt         # Pinned Python package dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Sachet.git
cd Sachet

# Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Key variables:
- `MODEL_PATH`: Path to local model folder (defaults to `models/sachet_muril_v2`).
- `HF_MODEL_REPO`: Your Hugging Face model repository ID (e.g., `username/sachet-muril-v2`).
- `USE_HF_INFERENCE`: Set to `true` for low-RAM serverless inference mode; `false` for local PyTorch execution.
- `SACHET_API_KEY`: Optional secret key to secure your API endpoints.

### 4. Running the Application Locally
```bash
# Start FastAPI backend (serves both API and frontend UI)
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at `http://127.0.0.1:8000` to interact with the dashboard.

---

## API Reference

### 1. Analyze Utterance
- **Endpoint**: `POST /v1/analyze`
- **Request Body**:
  ```json
  {
    "session_id": "session-101",
    "text": "Main CBI Cyber Cell se bol raha hoon, aapke naam par warrant hai.",
    "speaker": "caller"
  }
  ```
- **Response**:
  ```json
  {
    "risk_score": 52,
    "level": "medium",
    "reason": "Authority impersonation: Caller khud ko Police ya CBI adhikari bata raha hai.",
    "signals": ["Authority impersonation", "Arrest / legal threat"],
    "alert": false,
    "history": [18, 52]
  }
  ```

### 2. End Session
- **Endpoint**: `POST /v1/session/end`
- **Request Body**:
  ```json
  {
    "session_id": "session-101"
  }
  ```
- **Response**: `{"deleted": true}`

### 3. Service Health
- **Endpoint**: `GET /health`
- **Response**: `{"ok": true, "sessions": 0}`

---

## License

This project is licensed under the MIT License. Developed for defensive, research, and educational purposes to combat financial cyber fraud.
