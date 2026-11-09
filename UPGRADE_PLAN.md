# Cosimo Upgrade Plan — Fully Local, Zero Monthly Cost, Mac Mini Kiosk

Written 2026-11-09 after a full read of this repo and research into Mistral's Voxtral models.
This file is the brief for whoever (human or Claude Code) does the work on the Mac Mini.

## 0. Goal in one paragraph

Turn `cosimo-local` into the real Cosimo: a streaming, interruptible voice docent that runs
entirely on the Mac Mini with **no API keys and no per-minute bills**, using the full persona
from `persona.py`, a real language model instead of llama3.2:1b, streaming speech recognition
instead of batch Whisper, and a local voice instead of Microsoft edge-tts. When this is done,
delete the Vapi half of the repo.

## 1. First: find out what the Mac Mini is

Run this and write the result into section 7 at the bottom of this file before choosing models:

```bash
sysctl -n machdep.cpu.brand_string; sysctl -n hw.memsize | awk '{print $1/1073741824 " GB RAM"}'; sw_vers -productVersion
```

Model sizes below are keyed to unified memory. Everything else in the plan is the same.

## 2. What is wrong with the current code (verified findings)

| # | Problem | Where | Why it matters |
|---|---------|-------|----------------|
| 1 | "FREE local" edition is not local: edge-tts sends every reply to Microsoft | `client_local.py` TextToSpeech | Breaks when Microsoft changes things; data leaves the box |
| 2 | Local edition ignores `persona.py`; uses a 4-line prompt + llama3.2:1b | `LocalLLM._build_system_prompt` | Much dumber Cosimo than the Vapi one |
| 3 | Pipeline is fully sequential: record → batch STT → full LLM reply → full MP3 → afplay | `_run_conversation` | Latency is the sum of 5 stages; no barge-in |
| 4 | Silence detection is a raw amplitude threshold (0.01 mean abs) | `SpeechToText.listen_and_transcribe` | Museum ambient noise keeps it "speaking" up to the 30 s cap |
| 5 | Retrieval is naive keyword scoring over 377 items | `KnowledgeBase.search` | "the round painting with the baby" finds nothing |
| 6 | Wake-word CNN runs inside the audio callback every 80 ms on Python lists | `WakeWordDetector.listen` | Wasteful; risks audio overruns. `hey_cosimo.onnx` already exists and is unused |
| 7 | No interruption: `afplay` blocks until the whole reply has played | `TextToSpeech._play_audio` | Visitors cannot cut Cosimo off |
| 8 | README/.env.example describe Porcupine/Picovoice; code uses openwakeword + custom model | docs | Confusing for the next person |
| 9 | A file literally named `.env ` (trailing space) with real keys was committed | git history, cosimo2.0 | Removed from tree 2026-11-09; keys must be deleted in the Vapi dashboard |

## 3. Target architecture

```
Mic ──► Wake word (ONNX, own thread) ──► Pipecat pipeline (one process, streaming end to end)
                                          │
                                          ├─ VAD: Silero VAD (free, tiny) — real speech/silence detection
                                          ├─ STT: Voxtral-Mini-4B-Realtime-2602 via mlx-audio / voxmlx
                                          │        (Apache 2.0, streams at 480 ms delay, 13 languages)
                                          ├─ LLM: local model via MLX (mlx-lm) or Ollama — see §4
                                          │        system prompt = persona.py + full catalog index (~9k tokens, cached)
                                          │        tool: get_artwork(id) → full record from collection_optimized.json
                                          │        tool: search_collection(query) → hybrid BM25 + embedding search
                                          ├─ TTS: Kokoro-82M via mlx-audio (Apache 2.0, runs on Neural Engine)
                                          │        sentence-chunked streaming; cancel on barge-in
                                          └─ Speaker
```

Why Pipecat: it is the open-source voice pipeline framework with existing Apple-silicon
reference stacks using exactly these parts. It gives streaming between stages and barge-in for
free, and each stage is swappable (e.g. point TTS at a hosted cloned voice later without
touching the rest).

### Licensing constraints (do not violate)

- **Voxtral-Mini-4B-Realtime-2602**: Apache 2.0. OK for the museum.
- **Voxtral-4B-TTS-2603** (open weights): **CC BY-NC 4.0, non-commercial.** Do NOT ship it
  locally without a legal read. Use Kokoro instead. Hosted Voxtral TTS is paid; not in scope.
- **Voxtral 3** (anything with "3" in the name: Mini Transcribe 3, Small Transcribe 3, Mini TTS 3):
  API-only, metered. Not in scope.
- **Kokoro-82M**: Apache 2.0. OK.
- **Silero VAD**: MIT. OK.

## 4. Model sizing by Mac Mini memory

| Unified memory | STT | LLM (4-bit MLX) | TTS | Headroom |
|---|---|---|---|---|
| 16 GB | Voxtral Realtime 4-bit (~2.3 GB) | 7–8B class (Qwen3 8B / Llama 3.x 8B / Mistral 7B) ~5 GB | Kokoro | tight but OK |
| 24 GB | Voxtral Realtime 8-bit (~4.5 GB) | 12–14B class ~8 GB | Kokoro | comfortable |
| 32 GB | Voxtral Realtime 8-bit | 24B class (Mistral Small family) ~14 GB | Kokoro | comfortable |
| 48–64 GB | Voxtral Realtime 8-bit | 24–32B class ~15–20 GB, 30–50 tok/s | Kokoro | room for vision model + diarization |

Pick the largest LLM tier that leaves ≥ 6 GB free. Prefer MLX (`mlx-lm`) over Ollama for
tokens/sec on Apple silicon; Ollama is acceptable if MLX causes trouble.
Note for M1/M2 chips: use float16-based 4-bit Voxtral weights (no native bfloat16 on those GPUs).
M3 and later: bfloat16 variants.

## 5. Implementation phases (do them in order, each one shippable)

### Phase A — Hygiene (no model changes)
1. Confirm the old Vapi/Picovoice keys are deleted in their dashboards. `.gitignore` now blocks `.env*`.
2. Make `client_local.py` use `SYSTEM_PROMPT` and `FIRST_MESSAGE` from `persona.py`. Strip the
   Vapi-specific lines ("use the endCall function") into a small local variant that says to end
   with `[END_CONVERSATION]` instead.
3. Fix docs drift: README and `.env.example` should describe the local edition first; remove
   Porcupine/Picovoice references. Keep the Vapi docs in a short "legacy" section until Phase E.

### Phase B — Retrieval that works
1. Build `data/catalog_index.txt` at startup: one line per item, `id | title | artist | room | period`.
   377 items ≈ 9k tokens. Put it in the system prompt so Cosimo always knows what is in the building.
2. Add `get_artwork(id)` tool returning the full cleaned record (description, medium, culture,
   provenance, inscription, dimensions) from `collection_optimized.json`.
3. Replace keyword scoring with hybrid search: BM25 (`rank_bm25`) + local embeddings
   (`nomic-embed-text` via Ollama, or `sentence-transformers` all-MiniLM via MLX). Pre-compute
   embeddings once; cache to `data/embeddings.npy`.
4. Test queries that must work: "Giordano", "the round painting with the baby", "what's in the
   East Gallery", "do you have any furniture from France", "the Mona Lisa" (must say it's not here
   and redirect to a real item).

### Phase C — Streaming pipeline on Pipecat
1. New module `src/cosimo_vapi/pipeline.py` (or rename package to `cosimo`). Pipecat pipeline with:
   Silero VAD → Voxtral Realtime STT (mlx-audio) → LLM (mlx-lm or Ollama, streaming) →
   sentence aggregator → Kokoro TTS (mlx-audio) → speaker.
2. Barge-in: when VAD detects speech while TTS is playing, cancel TTS and the LLM stream.
3. End-of-turn: use VAD silence (~600–800 ms) rather than the 1.5 s amplitude timer.
4. Conversation end: `[END_CONVERSATION]` from the LLM, or goodbye phrases, or 3 unanswered turns.
5. Target: ≤ 1.0 s from end of visitor speech to first audio out. Log per-stage timings.

### Phase D — Wake word done properly
1. Load `hey_cosimo.onnx` through onnxruntime in its own thread; stop using the PyTorch `.pt`
   path at runtime (keep the training repo as is).
2. Feed it a rolling 1 s buffer from a `numpy` ring buffer, not Python lists.
3. Threshold in `.env` (`WAKE_THRESHOLD`, default 0.5). Log scores to tune false positives in situ.
4. Later: record 20–50 real "hey Cosimo" samples in the museum and retrain (see `WAKE_WORD_INTEGRATION.md`).

### Phase E — Remove the cloud
1. Delete `client.py`, `call_worker.py`, `setup_assistant.py`, the `vapi` extra in
   `pyproject.toml`, and the `cosimo` / `cosimo-setup` scripts. Keep `persona.py`, `test_audio.py`,
   `scripts/split_collection.py`.
2. Rename the package/project if desired (`cosimo` instead of `cosimo-vapi`).
3. Kiosk: `launchd` plist for auto-start on boot, `caffeinate` already handled, log rotation to `logs/`.

### Phase F — Extras the Mini's compute allows (optional, after A–E)
- Speaker diarization so Cosimo tracks who is talking in a group.
- A small local vision model + webcam so Cosimo knows which artwork the visitor is facing.
- Analytics: log questions (not audio) to a local SQLite file so the museum sees what people ask.
- Multilingual: Voxtral Realtime already handles 13 languages; Kokoro has non-English voices.

## 6. Dependencies to add (all free, all local)

```
pipecat-ai            # pipeline + barge-in
mlx, mlx-lm           # LLM on Apple silicon
mlx-audio[stt,tts]    # Voxtral Realtime STT + Kokoro TTS
silero-vad            # VAD
onnxruntime           # wake word (already installed)
rank-bm25             # lexical search
numpy, sounddevice    # already present
```

Remove: `faster-whisper`, `edge-tts`, `vapi-python`, `vapi-server-sdk`, `torch`/`torchaudio`
at runtime (keep only in the wake-word training repo).

## 7. Mac Mini hardware (fill in)

```
chip:
memory:
macOS:
```

## 8. Acceptance checklist

- [ ] `cosimo-local` runs with no network connection at all (turn Wi-Fi off and test)
- [ ] no API key of any kind in the repo or `.env`
- [ ] "Hey Cosimo" wakes it; a visitor can interrupt mid-sentence
- [ ] end of speech → first audio ≤ 1 s, logged per stage
- [ ] the five retrieval test queries in Phase B all pass
- [ ] persona matches `persona.py` (warm, brief, redirects absent works to real ones)
- [ ] survives 8 hours idle and auto-starts after a reboot
