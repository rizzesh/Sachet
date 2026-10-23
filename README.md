---
title: Sachet — Real-Time Scam Call Detector
emoji: 🛡️
colorFrom: red
colorTo: orange
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Hindi/Hinglish real-time scam call risk detector (MuRIL + heuristics)
---

# Sachet (सचेत) 🛡️

**Real-Time Scam-Call Risk Detector for Hindi & Hinglish Phone Calls**  
*Hack for Social Cause — VBYLD 2027 | Theme: Digital Safety & Cyber Fraud*  
*Defensive Use Only • "Warn, Never Block"*

---

## 📌 Overview

**Sachet** is an intelligent, privacy-first risk detection service engineered to safeguard citizens against evolving conversational cyber-frauds conducted in Hindi and Hinglish (e.g., Digital Arrest scams, fake KYC deactivations, custom duty extortion, and deceptive investment rackets).

Instead of relying on fragile keyword blacklists, Sachet inspects the **conversational arc and psychological escalation** (urgency induction, isolation tactics, authority impersonation, payment coercions) as speech-to-text turns are streamed. It outputs:
- **Risk Level**: `low`, `medium`, or `high`
- **Risk Score**: 0 to 100
- **Hindi Explanation (`reason_hi`)**: A clear, single-line explanation in simple Hindi warning the citizen about the specific deception tactic detected.

---

## 🔒 Privacy by Design

Sachet enforces strict privacy boundaries suited for sensitive phone call contexts:

1. **Text-Only Ingestion**:
   - Sachet **never** accepts, records, or processes raw audio.
   - Only textual transcripts (generated client-side via Web Speech API or local speech-to-text) are submitted to the API.
2. **Ephemeral In-Memory Storage**:
   - Active call session state (the evolving transcript and rolling risk window) resides **strictly in RAM**. No transcripts are persisted to a database or disk during analysis.
3. **Automatic Expiry (TTL)**:
   - Sessions expire and purge automatically after **30 minutes** of inactivity.
4. **Explicit Cleanup (`/v1/session/end`)**:
   - Calling the `/v1/session/end` endpoint immediately de-allocates and erases the entire session record from memory.
5. **No Data Retention for Training**:
   - User inputs sent to `/v1/analyze` are never retained or logged into persistent files.

---

## 🧭 Guiding Principle: "Warn, Never Block"

Sachet functions as an advisory co-pilot for the citizen, not an authoritarian firewall.
- **The user is always in control**: Calls are never forcibly terminated.
- **Clear Guidance**: Transparent reasoning in Hindi enables even non-tech-savvy or elderly users to independently recognize foul play and hang up.

---

## 📂 Repository Structure

```text
Sachet/
├── data/
│   ├── raw/             # Raw synthetic dialogues generated across scam & genuine grids
│   ├── processed/       # Deduplicated, validated, and turn-prefix expanded datasets
│   └── splits/          # Leakage-free train / validation / test splits partitioned by scenario
├── datagen/             # Scenario grid matrix & LLM dialogue generator
├── api/                 # FastAPI real-time analysis backend
├── client/              # Modern web demo interface (live transcript, meter, audio simulation)
├── eval/                # Evaluation suite (accuracy, F1, false-alarm rate, detection point)
├── notebooks/           # Colab/Kaggle-ready QLoRA / LoRA fine-tuning workflows
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation & execution guide
```

---

## 🚀 Phase Implementation Roadmap

- [x] **Phase 1: Setup** — Architecture blueprint, repository scaffold, dependencies, privacy spec.
- [ ] **Phase 2: Dataset Generation** — Scenario grid (7 scam types + 7 hard negatives, 4 victim profiles, 3 language modes) generating `data/raw/calls.jsonl`.
- [ ] **Phase 3: Data Processing** — Deduplication, scenario-based leakage-free splits, and progressive turn-prefix expansion in `data/processed/`.
- [ ] **Phase 4: Baseline Evaluation** — Few-shot context evaluator reporting precision, recall, hard-negative false-alarm rate, and detection point.
- [ ] **Phase 5: FastAPI Service** — Robust `/v1/analyze` and `/v1/session/end` endpoints with consecutive-update confirmation and in-memory TTL.
- [ ] **Phase 6: Web Client Demo** — Interactive dashboard with speech recognition, live transcript, risk meter, and sample scam replay.
- [ ] **Phase 7: Fine-Tuning** — Colab/Kaggle notebook fine-tuning multilingual models with LoRA/QLoRA against baseline.
- [ ] **Phase 8: Stretch Features** — Automated draft complaint for National Cybercrime Helpline 1930 & guardian alert triggers.
- [ ] **Phase 9: Final Polish** — Mermaid architecture diagrams, reproducibility benchmarks, and demonstration assets.

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### 2. Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example environment configuration and fill in your keys:
```powershell
Copy-Item .env.example .env
```
Edit `.env` to supply your preferred `LLM_API_KEY` (Gemini, OpenAI, or Groq) and custom `SACHET_API_KEY`.
