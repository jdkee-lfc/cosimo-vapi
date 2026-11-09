# CLAUDE.md

Cosimo is a voice museum docent for the Kreb's Center for the Humanities, running as a kiosk on a Mac Mini.

## Read first

`UPGRADE_PLAN.md` is the current brief. Follow its phases in order (A → E). Fill in section 7
(hardware) before choosing model sizes.

## Hard rules

- **Zero monthly cost.** No API keys, no metered services, nothing that phones home at runtime.
  edge-tts, Vapi, Picovoice, and all hosted Voxtral 3 models are out.
- **Licenses:** Voxtral-4B-TTS open weights are non-commercial — do not use locally. Use Kokoro.
  Voxtral-Mini-4B-Realtime (Apache 2.0) is fine.
- Never commit any file matching `.env*` other than `.env.example`. A misnamed `.env ` (trailing
  space) with real keys was committed once; `.gitignore` now blocks the pattern.
- Keep `persona.py` as the single source of Cosimo's personality.

## Layout

- `src/cosimo_vapi/client_local.py` — the local edition; this is what gets rebuilt
- `src/cosimo_vapi/client.py`, `call_worker.py`, `setup_assistant.py` — legacy Vapi cloud edition, delete in Phase E
- `data/collection_optimized.json` — 377 items (168 paintings, 124 sculptures, 70 furniture, 15 other)
- `hey_cosimo.onnx` / `hey_cosimo_weights.pt` — custom wake word model (ONNX is the one to use at runtime)
- `wake-word-training/` — nested git repo, not a submodule; not needed to run Cosimo

## Commands

```bash
source .venv/bin/activate
pip install -e ".[local]"
cosimo-local          # run the docent
cosimo-test-audio     # check mic and speaker
```
