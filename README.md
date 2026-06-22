# Cosimo — AI Museum Docent (Vapi Cloud Edition)

A voice-interactive museum docent powered by [Vapi.ai](https://vapi.ai). Cosimo listens for its wake word on a local Mac kiosk, then hands off the entire voice conversation to Vapi's cloud — which handles speech recognition, LLM reasoning with RAG over your museum collection, and natural text-to-speech. When the visitor stops talking, the call ends and Cosimo returns to listening.

```
┌─── Mac Studio (local) ───┐      ┌─── Vapi Cloud ──────────────┐
│                           │      │                              │
│  Microphone               │      │  Deepgram Nova-3 (STT)       │
│    │                      │      │         │                    │
│    ▼                      │      │         ▼                    │
│  Porcupine ("Cosimo")     │      │  GPT-4o + Knowledge Base     │
│    │                      │      │  (your 1200-item collection) │
│    ▼                      │      │         │                    │
│  vapi-python SDK ◄──WebRTC──►    │         ▼                    │
│    │                      │      │  ElevenLabs Flash v2.5 (TTS) │
│    ▼                      │      │                              │
│  Speaker                  │      └──────────────────────────────┘
│                           │
└───────────────────────────┘
```

**~200 lines of local Python. Everything else is cloud.**

## Setup (15 minutes)

### 1. Get API keys

| Service | URL | Cost |
|---------|-----|------|
| Vapi | [dashboard.vapi.ai](https://dashboard.vapi.ai/) | Pay-per-minute (~$0.05-0.15/min) |
| Picovoice | [console.picovoice.ai](https://console.picovoice.ai/) | Free tier available |

From the Vapi dashboard, you need both your **Private API Key** (for setup) and **Public Key** (for client calls).

### 2. Clone and install

```bash
git clone <your-repo> cosimo-vapi
cd cosimo-vapi
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On macOS you may need `brew install portaudio` first for PyAudio (dependency of vapi-python).

### 3. Configure

```bash
cp .env.example .env
nano .env
```

Fill in `VAPI_API_KEY`, `VAPI_PUBLIC_KEY`, and `PICOVOICE_ACCESS_KEY`.

### 4. Add your museum collection

Place your collection at `data/collection.json`. Format:

```json
{
  "items": [
    {
      "title": "Starry Night Over the Harbor",
      "artist": "Elena Vasquez",
      "date": "1987",
      "medium": "Oil on canvas",
      "description": "A luminous nightscape...",
      "gallery": "Gallery 3",
      "period": "Contemporary"
    }
  ]
}
```

Or copy the sample to test with: `cp data/sample_collection.json data/collection.json`

#### Optimizing large collections (recommended)

Vapi recommends keeping knowledge base files under 300KB for best retrieval performance. If your collection is larger, use the split script:

```bash
python scripts/split_collection.py
```

This creates optimized files by:
- Removing unnecessary fields (sourcePath, searchText, etc.)
- Splitting by category into smaller files

Output files:
| File | Contents |
|------|----------|
| `collection_paintings_galleries.json` | Paintings in gallery spaces |
| `collection_paintings_rooms.json` | Paintings in house rooms |
| `collection_sculptures.json` | All sculptures |
| `collection_furniture_other.json` | Furniture, textiles, decorative |
| `collection_optimized.json` | All items, cleaned (single file) |

The original `collection.json` is preserved.

### 5. Create the Vapi assistant

```bash
cosimo-setup
```

This uploads your collection, creates a knowledge base, builds the Cosimo assistant with persona and voice, and saves the assistant ID to `.env`. Run it once, or again with `--reset` to recreate from scratch.

#### Using split collection files

To use the optimized split files (recommended for large collections):

```bash
cosimo-setup --collection data/collection_paintings_galleries.json,data/collection_paintings_rooms.json,data/collection_sculptures.json,data/collection_furniture_other.json --reset
```

Or use the single optimized file:

```bash
cosimo-setup --collection data/collection_optimized.json --reset
```

The `--reset` flag creates a fresh assistant with the new files and updated persona.

### 6. Generate the wake word

1. Go to [console.picovoice.ai](https://console.picovoice.ai/)
2. Porcupine → Custom Keywords → type **"Cosimo"** → macOS arm64
3. Download `.ppn` to `data/cosimo_wake_word.ppn`

### 7. Test audio

```bash
cosimo-test-audio
```

### 8. Run

```bash
cosimo
```

Say **"Cosimo"** → conversation starts → silence ends it → back to listening.

## Project structure

```
cosimo-vapi/
├── .env                    # API keys (git-ignored)
├── pyproject.toml          # Dependencies
├── data/
│   ├── collection.json     # Your museum data (original)
│   ├── collection_*.json   # Split/optimized files (generated)
│   ├── sample_collection.json
│   └── cosimo_wake_word.ppn  # Porcupine model (git-ignored)
├── scripts/
│   └── split_collection.py # Split large collections for better RAG
└── src/cosimo_vapi/
    ├── client.py           # Main loop: wake word → Vapi call → repeat
    ├── setup_assistant.py  # One-time: upload collection → create assistant
    ├── persona.py          # Cosimo's system prompt
    └── test_audio.py       # Audio device test
```

## Customizing

### Change the voice

Edit `setup_assistant.py` and change the `voice` block. Vapi supports ElevenLabs, PlayHT, Deepgram, and others. To use a custom cloned voice from ElevenLabs, replace `voiceId` with your clone's ID.

### Change the persona

Edit `persona.py` and re-run `cosimo-setup` to push the updated prompt to Vapi.

### Change the LLM

Edit the `model` block in `setup_assistant.py`. Vapi supports OpenAI, Anthropic, Google, Groq, and others. For example, to use Claude:

```python
"model": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    ...
}
```

### Update the collection

Drop a new `data/collection.json` and re-run:

```bash
# If using split files (recommended for large collections)
python scripts/split_collection.py
cosimo-setup --collection data/collection_paintings_galleries.json,data/collection_paintings_rooms.json,data/collection_sculptures.json,data/collection_furniture_other.json --reset

# Or single file
cosimo-setup --reset
```

This uploads the new files and relinks the knowledge base.

## Kiosk deployment

For 24/7 operation, the same launchd approach from the local edition works:

```bash
# Prevent sleep
sudo pmset -a displaysleep 0 sleep 0 disksleep 0
sudo pmset -a autorestart 1

# Auto-start (create a LaunchDaemon plist pointing to the cosimo command)
```

The client auto-reconnects on errors and runs `caffeinate` to prevent macOS sleep.

## Cost estimate

For a museum open 8 hours/day where Cosimo averages 2 hours of active conversation:

| Component | Monthly cost |
|-----------|-------------|
| Vapi (~$0.10/min × 60 hrs) | ~$360 |
| Picovoice (free tier) | $0 |
| **Total** | **~$360/mo** |

If cost becomes a concern, the local Cosimo edition (from the earlier build) eliminates per-minute charges entirely.

## Compared to the local edition

| | Vapi Cloud | Local (Pipecat) |
|-|-----------|----------------|
| Setup time | 15 minutes | 2-4 hours |
| Local code | ~200 lines | ~2000 lines |
| Internet required | Yes (always) | Only for ElevenLabs TTS |
| Voice quality | Excellent (ElevenLabs via Vapi) | Excellent (same) |
| Latency | ~500-800ms (network dependent) | ~400-530ms |
| Monthly cost | ~$360 at 2hr/day active | ~$5 (ElevenLabs only) |
| Collection privacy | Data stored on Vapi servers | Fully local |
| Custom voice clone | Yes (via ElevenLabs) | Yes (via ElevenLabs) |
| Offline fallback | No | Yes (Kokoro TTS) |

Start with Vapi to validate the concept. Migrate to local when you've nailed the experience.
