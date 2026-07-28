# Cosimo Local Setup Guide

## Welcome!

This guide will help you set up **Cosimo**, our AI museum docent, using the **FREE local version**. This version runs entirely on your computer with no monthly costs.

---

## What You'll Need

- A Mac computer (Apple Silicon recommended, Intel works too)
- About **3GB of free disk space**
- A working microphone and speakers
- ~15 minutes for setup

---

## Step 1: Clone the Repository

If you haven't already, clone the repo and switch to the correct branch:

```bash
git clone <repository-url>
cd cosimo-vapi
git checkout cosimo2.0
git pull origin cosimo2.0
```

---

## Step 2: Install Ollama (The AI Brain)

Ollama is what powers Cosimo's intelligence. It runs AI models locally on your Mac.

### Option A: Using Homebrew (Recommended)
```bash
brew install ollama
```

### Option B: Direct Download
1. Go to [https://ollama.ai](https://ollama.ai)
2. Click "Download for macOS"
3. Open the downloaded file and drag Ollama to Applications
4. Open Ollama from Applications (it will run in your menu bar)

### Download the AI Model

After installing Ollama, download the language model:

```bash
ollama pull llama3.2:1b
```

This downloads a ~1.3GB model. It only needs to happen once.

> **Note:** If you have more disk space (3GB+), you can use the smarter model:
> ```bash
> ollama pull llama3.2
> ```

---

## Step 3: Set Up Python Environment

### Install portaudio (required for microphone access)
```bash
brew install portaudio
```

### Create and activate virtual environment
```bash
cd cosimo-vapi
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -e ".[local]"
```

This installs:
- **faster-whisper** - Speech recognition (converts your voice to text)
- **ollama** - AI language model interface
- **edge-tts** - Text-to-speech (Cosimo's voice)

---

## Step 4: Configure Environment Variables

Create your `.env` file:

```bash
cp .env.example .env
```

Open `.env` in a text editor and make sure these lines are present:

```
# Wake word detection
WAKE_WORD=hey_cosimo
WAKE_WORD_MODEL=./hey_cosimo_weights.pt

# Local FREE version settings
OLLAMA_MODEL=llama3.2:1b
TTS_VOICE=en-US-GuyNeural
```

> **Voice Options:** You can change `TTS_VOICE` to:
> - `en-US-GuyNeural` - Male American voice (default)
> - `en-US-JennyNeural` - Female American voice
> - `en-GB-RyanNeural` - Male British voice
> - `en-GB-SoniaNeural` - Female British voice

---

## Step 5: Run Cosimo!

Make sure Ollama is running (check your menu bar for the Ollama icon), then:

```bash
source .venv/bin/activate
cosimo-local
```

You should see:

```
╔══════════════════════════════════════════╗
║  COSIMO — Museum Docent (FREE Local)     ║
╚══════════════════════════════════════════╝

  LLM Model: llama3.2:1b
  TTS Voice: en-US-GuyNeural
  Wake word: hey cosimo

Listening for wake word...  Say 'Hey Cosimo'
```

### How to Use

1. Say **"Hey Cosimo"** to wake up the assistant
2. Ask questions about the museum collection
3. Say **"goodbye"** or **"that's all"** to end the conversation
4. Cosimo will return to listening mode

---

## Troubleshooting

### "Ollama not running or not installed"

Make sure Ollama is running:
```bash
ollama serve
```
Or open the Ollama app from Applications.

### "No space left on device"

The AI model needs ~1.3GB of space. Free up disk space by:
```bash
# Empty trash
rm -rf ~/.Trash/*

# Clear caches
rm -rf ~/Library/Caches/pip/*
rm -rf ~/Library/Caches/Homebrew/*
```

### Slow response times

The local version is optimized for speed, but if it's still slow:
1. Close other heavy applications
2. Make sure you're using `llama3.2:1b` (smaller, faster model)

### Microphone not working

Test your audio devices:
```bash
cosimo-test-audio
```

---

## How It Works (Technical Overview)

| Component | Technology | Cost |
|-----------|------------|------|
| Wake Word Detection | Custom PyTorch model | FREE |
| Speech-to-Text | faster-whisper (local) | FREE |
| AI Brain | Ollama + Llama 3.2 | FREE |
| Text-to-Speech | edge-tts (Microsoft) | FREE |

**Total monthly cost: $0**

All processing happens on your computer. No data is sent to external servers (except for the text-to-speech which uses Microsoft's free Edge TTS service).

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `cosimo-local` | Run the FREE local version |
| `cosimo` | Run the Vapi cloud version (requires paid API) |
| `cosimo-test-audio` | Test microphone and speakers |
| `cosimo-setup` | Set up Vapi assistant (cloud version only) |

---

## Questions?

If you run into issues, check:
1. Is Ollama running? (Look for icon in menu bar)
2. Did you activate the virtual environment? (`source .venv/bin/activate`)
3. Is your microphone working? (`cosimo-test-audio`)

Happy docent-ing! 🎨
