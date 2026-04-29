# Wake Word Integration & Call Improvements

## Overview

This update adds a custom-trained "Hey Cosimo" wake word model and improves the call experience with better conversation handling.

---

## Changes Made

### 1. Custom Wake Word Model

- **Trained locally** using Edge TTS to generate synthetic voice samples
- **368 training samples** (244 positive "hey cosimo", 124 negative phrases)
- **95.95% validation accuracy**
- Model file: `hey_cosimo_weights.pt` (534KB)

**Files added:**
- `hey_cosimo_weights.pt` — trained PyTorch model weights
- `wake-word-training/` — training pipeline and scripts

**Files modified:**
- `src/cosimo_vapi/client.py` — added `CustomWakeWordModel` class and PyTorch model loading
- `.env` — added `WAKE_WORD` and `WAKE_WORD_MODEL` settings

### 2. Automatic End-Call Phrases

Cosimo now ends the call automatically when visitors say:
- "goodbye cosimo" / "goodbye" / "bye cosimo"
- "that's all" / "that's all thanks"
- "i'm done" / "thanks bye" / "end call"

**Files modified:**
- `src/cosimo_vapi/setup_assistant.py` — added `endCallPhrases` configuration

### 3. Improved Conversation Handling

Cosimo no longer interrupts side conversations between visitors:
- Only responds when directly addressed or asked a question
- Stays silent during overheard conversations
- No more filler phrases like "just a moment" or "let me check"

**Files modified:**
- `src/cosimo_vapi/persona.py` — added "ONLY RESPOND WHEN ADDRESSED" rules

---

## How to Run

### Prerequisites

```bash
# Python 3.12 recommended
python --version

# Install dependencies
cd /path/to/cosimo-vapi
source .venv/bin/activate
pip install torch torchaudio
```

### Start Cosimo

```bash
source .venv/bin/activate
python -m cosimo_vapi.client
```

Then say **"Hey Cosimo"** to start a conversation.

### Update Assistant (after persona changes)

```bash
source .venv/bin/activate
python -m cosimo_vapi.setup_assistant
```

---

## Environment Variables

Add these to your `.env` file:

```env
# Wake word detection
WAKE_WORD=hey_cosimo
WAKE_WORD_MODEL=./hey_cosimo_weights.pt
```

---

## Future Recommendations

### 1. Reduce Response Latency

**Problem:** The collection file is 1.3MB (Vapi recommends <300KB), causing slow knowledge searches.

**Solutions:**
- Split `data/collection.json` into smaller category-based files (e.g., `paintings.json`, `sculptures.json`)
- Create a condensed version with only essential fields (title, artist, date, description)
- Use Vapi's chunking settings to optimize retrieval

### 2. Improve Wake Word Accuracy

**Current:** 95.95% accuracy with synthetic voices only.

**To improve:**
- Record 20-50 real voice samples saying "hey cosimo" in the museum environment
- Add ambient museum noise to training data
- Retrain with mixed synthetic + real samples
- Adjust detection threshold in `.env` if false positives/negatives occur

**Recording samples:**
```bash
cd wake-word-training
source venv/bin/activate
python record_samples.py --wake-word "hey cosimo"
```

### 3. Reduce False Activations

If Cosimo activates on similar-sounding words:
- Increase threshold: edit `client.py` line with `threshold=0.5` → `threshold=0.7`
- Add more negative training samples for confusing words ("hey cosmos", "casino", etc.)

### 4. Faster Model Inference

Current model runs on CPU. For faster inference:
- Use Apple's MPS (Metal Performance Shaders) — already supported in code
- Quantize the model to reduce size and speed up inference
- Consider ONNX Runtime for optimized inference

### 5. Handle Network Latency

For slow Vapi responses:
- Enable response streaming in Vapi dashboard
- Use a faster TTS voice (e.g., `eleven_flash_v2_5` already configured)
- Consider edge caching for common questions

### 6. Museum-Specific Improvements

- **Proximity detection:** Add ultrasonic/IR sensor to only activate when someone is nearby
- **Multi-language support:** Train wake words for other languages
- **Quiet hours mode:** Reduce sensitivity during low-traffic times
- **Analytics:** Log popular questions to improve collection descriptions

---

## Troubleshooting

### Wake word not detecting
- Check microphone permissions
- Verify `WAKE_WORD_MODEL` path in `.env`
- Try lowering threshold to `0.3` for testing

### Call not ending on "goodbye"
- Run `python -m cosimo_vapi.setup_assistant` to update the assistant
- Check Vapi dashboard for `endCallPhrases` configuration

### Cosimo still interrupting side conversations
- Run setup again to push latest persona changes
- The LLM-based filtering isn't perfect — this is a limitation of the approach

### Slow responses
- Check your internet connection
- Consider splitting the collection file (see recommendation #1)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Local (Mac Studio)                   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  Microphone     │───▶│  Wake Word Detector         │ │
│  │                 │    │  (CustomWakeWordModel)      │ │
│  └─────────────────┘    └───────────┬─────────────────┘ │
│                                     │ "hey cosimo"      │
│                                     ▼                   │
│                         ┌─────────────────────────────┐ │
│                         │  Vapi Call (subprocess)    │ │
│                         │  - Audio streaming          │ │
│                         │  - Real-time conversation   │ │
│                         └───────────┬─────────────────┘ │
└─────────────────────────────────────┼───────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────┐
│                    Cloud (Vapi)                         │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Deepgram    │  │  GPT-4o      │  │  ElevenLabs  │  │
│  │  (STT)       │─▶│  + Knowledge │─▶│  (TTS)       │  │
│  │              │  │  Base Search │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Files Overview

```
cosimo-vapi/
├── .env                          # API keys and wake word config
├── hey_cosimo_weights.pt         # Trained wake word model
├── src/cosimo_vapi/
│   ├── client.py                 # Main client with wake word detection
│   ├── persona.py                # Cosimo's personality and rules
│   ├── setup_assistant.py        # Vapi assistant configuration
│   └── call_worker.py            # Subprocess for Vapi calls
├── data/
│   └── collection.json           # Museum collection data
└── wake-word-training/           # Training pipeline
    ├── train-lightweight.py      # Mac-compatible training script
    ├── best_model.pt             # Trained model checkpoint
    └── training_data/            # Generated audio samples
```
