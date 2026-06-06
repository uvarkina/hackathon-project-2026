# 🛡️ SafeWawe

**A quiet safety layer that listens during phone calls and warns the user the moment a scam starts to look real.**

SafeWawe runs in the background of a call and combines two signals in real time:

1. **AI-voice detection** — is the caller a synthetic / cloned voice?
2. **Fraud-phrase detection** — is the caller using known scam scripts (bank, police, tax authority, "you won a prize", etc.)?

When the combined risk rises, a calm companion card appears, explains the danger, suggests what to say, and — above a high threshold — can vibrate the phone and notify a trusted family member. It never takes over the call.

> Built for Hebrew-language scam calls, but the pipeline is language-agnostic.

---

## ✨ Features

- **Live transcription** of the call with risky phrases highlighted as they appear (faster-whisper).
- **Real-time AI-voice scoring** every few seconds, with temporal smoothing and a silence gate.
- **Scaling alert** — icon, color and percentage grow with the risk so the user notices without panic.
- **Family support** — above 85 % risk, one tap notifies a trusted contact (Twilio WhatsApp).
- **Pluggable detection engine** — switch between a fine-tuned model, your own trained model, or a fast heuristic with one env variable.

---

## 🧠 How it works

```
┌─────────────┐   WAV chunks (≈3 s)    ┌───────────────────────┐
│  Frontend   │ ───────ws/stream─────► │  Backend (FastAPI)     │
│ index.html  │ ◄──────JSON score───── │  port 8000             │
└─────────────┘                        │                        │
                                       │  • analysis.py         │  AI-voice score
                                       │    (voice engine)      │
                                       │  • calls NLP service   │
                                       └──────────┬─────────────┘
                                                  │ HTTP / direct call
                                       ┌──────────▼─────────────┐
                                       │  NLP service           │  transcript +
                                       │  text_fraud_recognition│  fraud-phrase score
                                       │  port 8001             │  (faster-whisper)
                                       └────────────────────────┘
```

The browser captures microphone audio, encodes it to WAV, and streams ~3-second chunks over a WebSocket. The backend scores each chunk for synthetic voice **and** transcribes it to match against fraud phrases, then returns a combined risk level (`safe` / `warning` / `danger` / `alert`).

---

## 🗂️ Project structure

```
backend/
  main.py            FastAPI app, WebSocket /ws/stream, risk fusion, alerts
  analysis.py        Voice engine: pretrained / own model / heuristic (+ silence gate, smoothing)
  notifier.py        Twilio WhatsApp alert
frontend/
  index.html         SafeWawe UI (served at http://localhost:8000)
  static/icon.png    Logo
text_fraud_recognition/
  app.py             NLP microservice (port 8001)
  transcriber.py     faster-whisper transcription (forced Hebrew)
  fraud_detector.py  fraud-phrase matching
  fraud_phrases.json Hebrew scam phrases
retrain_model.py        Train the in-house Conformer model
finetune_wav2vec2.py    Fine-tune a pretrained wav2vec2 detector on your data  ← recommended
gen_hebrew_ai.py        Generate Hebrew AI voice samples with gTTS
check_model.py          Test the Conformer model on audio files
check_spoof.py          Test the pretrained / fine-tuned model on audio files
data/
  ai/      drop AI / synthetic voice clips here   (training, gitignored)
  human/   drop real human voice clips here       (training, gitignored)
requirements.txt
start.sh                Launches both services and opens the browser
.env.example            Copy to .env and fill in
```

---

## 🚀 Setup

> **Use Python 3.12.** TensorFlow has no wheels for 3.14, and the macOS system Python can crash on import. A clean python.org / Homebrew 3.12 in a virtualenv avoids all of it.

```bash
# 1. Create and activate a virtual environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Open `.env` and set the detection engine (defaults are sensible):

```ini
VOICE_ENGINE=pretrained                 # pretrained | model | heuristic
SPOOF_MODEL=uvarkina/safewawe-hebrew    # our fine-tuned model on HuggingFace (auto-downloaded)
MODEL_OUTPUT_IS_AI=1                    # only for VOICE_ENGINE=model (retrained Conformer)
```

Twilio keys are only needed for WhatsApp alerts — the detector works without them.

---

## ▶️ Running

```bash
source .venv/bin/activate
./start.sh
```

This starts:

- **Backend / UI** → http://localhost:8000
- **NLP service** → http://localhost:8001
- **Call history** → http://localhost:8000/history

Open the page, press the call button, and speak. Stop everything with **Ctrl + C**.

> First run downloads the Whisper model and (for `pretrained`) the wav2vec2 weights. If the UI looks stale after an update, hard-refresh with **Cmd + Shift + R**.

---

## 🎚️ Detection engines

Set `VOICE_ENGINE` in `.env`:

| Value        | What it uses                              | Best for |
|--------------|-------------------------------------------|----------|
| `pretrained` | wav2vec2 deepfake detector, optionally fine-tuned on your data (`wav2vec2_finetuned/`) | **Recommended** — robust on real human voice |
| `model`      | In-house Conformer (`conformer_audio_model.keras`) | Lightweight, fully self-trained |
| `heuristic`  | librosa MFCC/bandwidth formula            | No ML deps, fallback only |

All engines are loaded lazily and crash-proof: if a model is missing, the backend automatically falls back instead of failing to start.

---

## 🧪 Training on your own data

The detector is only as good as its data. Both training scripts auto-load any audio you drop into the data folders — **no code edits needed**:

```
data/ai/      → AI / synthetic voice   (label: AI)
data/human/   → real human voice       (label: human)
```
Supported formats: `.mp3 .wav .flac .m4a .ogg .webm .aac`.

### 1. Generate Hebrew AI samples (optional)

```bash
pip install gTTS
python gen_hebrew_ai.py        # gTTS → data/ai/
```

### 2a. Fine-tune the pretrained model (recommended)

```bash
python finetune_wav2vec2.py    # → wav2vec2_finetuned/
```
Then set in `.env`: `VOICE_ENGINE=pretrained` and `SPOOF_MODEL=./wav2vec2_finetuned`.

### 2b. …or retrain the in-house Conformer

```bash
python retrain_model.py        # → conformer_audio_model.keras
```
Then set in `.env`: `VOICE_ENGINE=model` and `MODEL_OUTPUT_IS_AI=1`.

### 3. Test

```bash
python check_model.py data/human    # in-house model
python check_spoof.py data/human    # pretrained / fine-tuned model
```

> **Tip — match the channel.** The model must hear the same audio path it will see live. To collect microphone audio exactly as the site captures it, run `CAPTURE_DIR=data/human ./start.sh`, talk for a few minutes, then `Ctrl + C` and retrain.

---

## ⚠️ Known limitations

- **Data scarcity / domain gap.** Off-the-shelf deepfake detectors are English-trained and miss Hebrew synthetic voices; a model trained on too few / too-homogeneous samples flips between false positives (real voice flagged as AI) and false negatives (AI missed). More diverse Hebrew data — multiple voices, TTS engines, microphones and recording conditions — is the single biggest lever.
- **One TTS = one fingerprint.** gTTS produces a single voice. For robustness, include samples from the TTS engines you actually expect to face.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|--------|-----|
| `No matching distribution found for tensorflow` | You're on Python 3.14. Use a **3.12** venv. |
| `mutex lock failed` on `import tensorflow` | macOS system Python. Use a python.org / Homebrew 3.12 venv. |
| `ModuleNotFoundError` although `pip` said "already satisfied" | `pip` and `python` point to different interpreters — activate the venv. |
| UI shows the old design after an update | Browser cache — **Cmd + Shift + R**, or open in a private window. |
| Live mic flagged as AI but recordings aren't | Channel mismatch — capture live-mic audio (`CAPTURE_DIR=data/human ./start.sh`) and retrain. |

---

## 📦 Tech stack

FastAPI · WebSockets · PyTorch · TensorFlow / Keras · 🤗 Transformers (wav2vec2) · faster-whisper · librosa · Twilio · vanilla-JS frontend.
