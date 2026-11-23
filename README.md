# Cosimo — Voice Museum Docent

Cosimo is a voice docent for the Kreb's Center for the Humanities. It runs as a kiosk on a Mac
Mini, entirely on the machine: **no API keys, no monthly cost, nothing phones home.**

Say **"Hey Cosimo"**, ask about anything in the collection (377 items: paintings, sculptures,
furniture), and say "goodbye" or "that's all" when you're done.

> **Work in progress.** The local edition is being rebuilt according to
> [`UPGRADE_PLAN.md`](UPGRADE_PLAN.md). Check the plan to see which phase is current.

## Target architecture

```
Mic ──► Wake word "Hey Cosimo" (hey_cosimo.onnx)
          │
          ▼
        Pipecat pipeline (streaming, interruptible)
          ├─ VAD   Silero VAD
          ├─ STT   Voxtral-Mini-4B-Realtime (mlx-audio, Apache 2.0)
          ├─ LLM   local model via MLX / Ollama, persona from persona.py,
          │        tools over data/collection_optimized.json
          ├─ TTS   Kokoro-82M (mlx-audio, Apache 2.0)
          ▼
        Speaker
```

The current interim code (`client_local.py`) still uses faster-whisper, Ollama, and edge-tts.
edge-tts sends text to Microsoft and will be replaced by Kokoro in Phase C.

## Setup

Requirements: an Apple-silicon Mac, a microphone and speaker, and [uv](https://docs.astral.sh/uv/)
(for Python 3.12).

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[local]"

cp .env.example .env        # settings only, no keys; edit as needed
cosimo-test-audio           # check mic and speaker
cosimo-local                # run the docent
```

The interim edition also needs [Ollama](https://ollama.com) running, with a model pulled
(`LLM_MODEL` in `.env`, default `llama3.2`).

## Configuration

Everything lives in `.env`, and `.env.example` documents every setting. It never holds an API
key. `.gitignore` blocks every `.env*` file except `.env.example`.

## Project layout

```
src/cosimo_vapi/
  persona.py          Cosimo's personality — the single source for every edition
  client_local.py     local edition (being rebuilt)
  test_audio.py       audio device check
  client.py, call_worker.py, setup_assistant.py   legacy Vapi edition (removed in Phase E)
data/
  collection_optimized.json   the 377-item collection
hey_cosimo.onnx               wake word model (runtime)
hey_cosimo_weights.pt         same model as PyTorch weights (training)
scripts/split_collection.py   cleans/splits the raw collection export
UPGRADE_PLAN.md               the rebuild brief
WAKE_WORD_INTEGRATION.md      how the wake word was trained
```

## Legacy: Vapi cloud edition

The first version handed each conversation to Vapi's cloud (Deepgram, GPT-4o, ElevenLabs) at
roughly $0.10 per minute. It's still in the repo (`cosimo`, `cosimo-setup` commands, `vapi` extra)
but is no longer used. It will be deleted in Phase E of the upgrade plan, and any old Vapi or
Picovoice keys should be considered revoked.
