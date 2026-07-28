"""Cosimo local client — 100% FREE version using local/free services.

This version replaces Vapi with:
  - STT: faster-whisper (local Whisper model)
  - LLM: Ollama (local Llama 3.1 or similar)
  - TTS: edge-tts (free Microsoft TTS)

State machine:
  IDLE → [wake word detected] → ACTIVE (conversation) → [silence/goodbye] → IDLE
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from dotenv import load_dotenv
from rich.console import Console

console = Console()

# Audio settings
SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # 80ms at 16kHz
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5  # seconds of silence to end turn (faster response)

# Lazy imports for optional heavy dependencies
whisper_model = None
ollama = None


def get_whisper():
    """Lazy load faster-whisper."""
    global whisper_model
    if whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            console.print("  [dim]Loading Whisper model...[/]")
            # Use "tiny" for fastest speed (3x faster than base)
            whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            console.print("  [dim]Whisper loaded.[/]")
        except ImportError:
            console.print("[red]faster-whisper not installed. Run: pip install faster-whisper[/]")
            sys.exit(1)
    return whisper_model


def get_ollama():
    """Lazy load ollama."""
    global ollama
    if ollama is None:
        try:
            import ollama as _ollama
            ollama = _ollama
        except ImportError:
            console.print("[red]ollama not installed. Run: pip install ollama[/]")
            sys.exit(1)
    return ollama


# ---------------------------------------------------------------------------
# Custom wake word model (reused from client.py)
# ---------------------------------------------------------------------------

class CustomWakeWordModel(torch.nn.Module):
    """Simple CNN model for wake word detection."""

    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=160, n_mels=40
        )
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.mel_spec(x)
        x = torch.log(x + 1e-9)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# Wake word detector (reused from client.py)
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Listens continuously for wake word using custom model."""

    def __init__(self, wake_word: str = "hey_cosimo", model_path: str | None = None, threshold: float = 0.5):
        self.wake_word = wake_word
        self.model_path = model_path
        self.threshold = threshold
        self._custom_model = None
        self._initialized = False
        self._audio_buffer = []

    def _init(self):
        if self._initialized:
            return

        console.print("  [dim]Loading wake word model...[/]")

        if self.model_path and self.model_path.endswith('.pt') and Path(self.model_path).exists():
            console.print(f"  [dim]Using custom model: {self.model_path}[/]")
            self._custom_model = CustomWakeWordModel()
            self._custom_model.load_state_dict(
                torch.load(self.model_path, weights_only=True, map_location='cpu')
            )
            self._custom_model.eval()
        else:
            console.print("[yellow]No wake word model found. Using always-on mode.[/]")

        self._initialized = True
        console.print(f"  Wake word: [bold]{self.wake_word.replace('_', ' ')}[/]")

    def listen(self) -> bool:
        """Block until wake word is detected. Returns True on detection."""
        self._init()

        if self._custom_model is None:
            # No model - just wait for user to press Enter
            input("  [Press Enter to start conversation]")
            return True

        detected = False
        self._audio_buffer = []

        def callback(indata, frames, time_info, status):
            nonlocal detected
            if status or detected:
                return

            self._audio_buffer.extend(indata[:, 0].tolist())

            if len(self._audio_buffer) > 16000:
                self._audio_buffer = self._audio_buffer[-16000:]

            if len(self._audio_buffer) >= 16000:
                audio = torch.FloatTensor(self._audio_buffer[-16000:]).unsqueeze(0)
                with torch.no_grad():
                    prob = self._custom_model(audio).item()
                if prob > self.threshold:
                    detected = True

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            callback=callback,
        )

        with stream:
            sd.sleep(500)
            detected = False
            self._audio_buffer = []
            while not detected:
                sd.sleep(100)

        return True

    def cleanup(self):
        self._custom_model = None
        self._initialized = False


# ---------------------------------------------------------------------------
# Speech-to-Text using faster-whisper
# ---------------------------------------------------------------------------

class SpeechToText:
    """Records audio and transcribes using faster-whisper."""

    def __init__(self):
        self._model = None

    def _init(self):
        if self._model is None:
            self._model = get_whisper()

    def listen_and_transcribe(self, timeout: float = 10.0) -> str | None:
        """Record audio until silence, then transcribe."""
        self._init()

        console.print("  [dim]Listening...[/]")

        audio_chunks = []
        silence_start = None
        is_speaking = False
        start_time = time.time()

        def callback(indata, frames, time_info, status):
            nonlocal silence_start, is_speaking
            if status:
                return

            audio_chunks.append(indata.copy())

            # Check audio level
            level = np.abs(indata).mean()

            if level > SILENCE_THRESHOLD:
                is_speaking = True
                silence_start = None
            elif is_speaking:
                if silence_start is None:
                    silence_start = time.time()

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            callback=callback,
        )

        with stream:
            # Wait for speech or timeout
            while True:
                sd.sleep(100)

                # Timeout waiting for speech
                if not is_speaking and (time.time() - start_time) > timeout:
                    console.print("  [dim]No speech detected[/]")
                    return None

                # End after silence
                if silence_start and (time.time() - silence_start) > SILENCE_DURATION:
                    break

                # Max recording time
                if (time.time() - start_time) > 30:
                    break

        if not audio_chunks:
            return None

        # Combine audio and transcribe
        audio = np.concatenate(audio_chunks, axis=0).flatten()

        if len(audio) < SAMPLE_RATE:  # Less than 1 second
            return None

        console.print("  [dim]Transcribing...[/]")

        # Use beam_size=1 for fastest transcription
        segments, _ = self._model.transcribe(audio, beam_size=1, language="en")
        text = " ".join([segment.text for segment in segments]).strip()

        if text:
            console.print(f"  [cyan]You said:[/] {text}")

        return text if text else None


# ---------------------------------------------------------------------------
# Text-to-Speech using edge-tts (free Microsoft TTS)
# ---------------------------------------------------------------------------

class TextToSpeech:
    """Converts text to speech using edge-tts (free)."""

    def __init__(self, voice: str = "en-US-GuyNeural"):
        self.voice = voice
        self._temp_dir = tempfile.mkdtemp()

    def speak(self, text: str):
        """Convert text to speech and play it."""
        if not text:
            return

        console.print(f"  [green]Cosimo:[/] {text}")

        # Generate audio file
        output_file = os.path.join(self._temp_dir, "speech.mp3")

        # Run edge-tts
        asyncio.run(self._generate_speech(text, output_file))

        # Play the audio
        self._play_audio(output_file)

    async def _generate_speech(self, text: str, output_file: str):
        """Generate speech using edge-tts."""
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_file)
        except ImportError:
            console.print("[red]edge-tts not installed. Run: pip install edge-tts[/]")
            return

    def _play_audio(self, file_path: str):
        """Play audio file using system command."""
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", file_path], capture_output=True)
        elif sys.platform == "linux":
            # Try mpv, then ffplay, then aplay
            for player in ["mpv --no-video", "ffplay -nodisp -autoexit", "aplay"]:
                try:
                    subprocess.run(player.split() + [file_path], capture_output=True)
                    break
                except FileNotFoundError:
                    continue
        elif sys.platform == "win32":
            # Windows Media Player
            os.startfile(file_path)

    def cleanup(self):
        """Clean up temp files."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir)
        except:
            pass


# ---------------------------------------------------------------------------
# Local Knowledge Base (simple RAG)
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """Simple local knowledge base for museum collection."""

    def __init__(self, collection_path: str = "data/collection_optimized.json"):
        self.collection_path = collection_path
        self.items = []
        self._loaded = False

    def load(self):
        """Load collection from JSON file."""
        if self._loaded:
            return

        path = Path(self.collection_path)
        if not path.exists():
            # Try other locations
            for alt in ["data/collection.json", "collection.json"]:
                if Path(alt).exists():
                    path = Path(alt)
                    break

        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.items = data.get("items", [])
            console.print(f"  [dim]Loaded {len(self.items)} items from collection[/]")
        else:
            console.print("[yellow]No collection file found. Cosimo won't have artwork knowledge.[/]")

        self._loaded = True

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        """Simple keyword search through collection."""
        self.load()

        if not self.items:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_items = []
        for item in self.items:
            score = 0
            searchable = " ".join([
                str(item.get("title", "")),
                str(item.get("artistName", "")),
                str(item.get("description", "")),
                str(item.get("period", "")),
                str(item.get("medium", "")),
                str(item.get("culture", "")),
                str(item.get("room", "")),
            ]).lower()

            # Score based on word matches
            for word in query_words:
                if len(word) > 2 and word in searchable:
                    score += 1
                    # Bonus for title/artist match
                    if word in str(item.get("title", "")).lower():
                        score += 2
                    if word in str(item.get("artistName", "")).lower():
                        score += 2

            if score > 0:
                scored_items.append((score, item))

        # Sort by score and return top results
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:max_results]]

    def format_item(self, item: dict) -> str:
        """Format an item for the LLM context."""
        parts = []
        if item.get("title"):
            parts.append(f"Title: {item['title']}")
        if item.get("artistName"):
            parts.append(f"Artist: {item['artistName']}")
        if item.get("period"):
            parts.append(f"Period: {item['period']}")
        if item.get("medium"):
            parts.append(f"Medium: {item['medium']}")
        if item.get("culture"):
            parts.append(f"Culture: {item['culture']}")
        if item.get("room"):
            parts.append(f"Location: {item['room']}")
        if item.get("description"):
            # Truncate long descriptions
            desc = item['description'][:500] + "..." if len(item['description']) > 500 else item['description']
            parts.append(f"Description: {desc}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM using Ollama (local)
# ---------------------------------------------------------------------------

class LocalLLM:
    """Chat with local LLM using Ollama."""

    def __init__(self, model: str = "llama3.2", knowledge_base: KnowledgeBase = None):
        self.model = model
        self.knowledge_base = knowledge_base
        self.conversation_history = []
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """You are Cosimo, a friendly museum guide at the Kreb's Center.

RULES:
- Be VERY brief: 1 sentence only, max 20 words.
- Speak naturally, no bullet points.
- Use artwork info if provided. Don't invent details.
- On goodbye/thanks, say brief farewell + [END_CONVERSATION]"""

    def chat(self, user_message: str) -> tuple[str, bool]:
        """Send message to LLM and get response. Returns (response, should_end)."""
        ollama_client = get_ollama()

        # Search knowledge base for relevant context
        context = ""
        if self.knowledge_base:
            results = self.knowledge_base.search(user_message)
            if results:
                context = "\n\nRelevant artwork information:\n"
                for item in results:
                    context += "\n---\n" + self.knowledge_base.format_item(item)

        # Build messages
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add conversation history (keep last 10 exchanges)
        messages.extend(self.conversation_history[-20:])

        # Add current message with context
        full_message = user_message
        if context:
            full_message = f"{user_message}\n{context}"

        messages.append({"role": "user", "content": full_message})

        try:
            response = ollama_client.chat(
                model=self.model,
                messages=messages,
                options={"num_predict": 100}  # Limit response length for speed
            )
            assistant_message = response['message']['content']

            # Check for end conversation marker
            should_end = "[END_CONVERSATION]" in assistant_message
            assistant_message = assistant_message.replace("[END_CONVERSATION]", "").strip()

            # Also check for goodbye phrases
            goodbye_phrases = ["goodbye", "bye", "farewell", "take care", "see you"]
            user_lower = user_message.lower()
            if any(phrase in user_lower for phrase in goodbye_phrases):
                should_end = True

            # Update history (without context to save tokens)
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            return assistant_message, should_end

        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/]")
            return "I'm sorry, I'm having trouble thinking right now. Could you try again?", False

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class CosimoLocal:
    """Top-level controller: wake word → conversation → repeat."""

    def __init__(self):
        load_dotenv()

        self.wake_word = os.getenv("WAKE_WORD", "hey_cosimo")
        self.wake_word_model = os.getenv("WAKE_WORD_MODEL", "./hey_cosimo_weights.pt")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.tts_voice = os.getenv("TTS_VOICE", "en-US-GuyNeural")
        self.collection_path = os.getenv("COLLECTION_PATH", "data/collection_optimized.json")

        self.wake_detector: WakeWordDetector | None = None
        self.stt: SpeechToText | None = None
        self.tts: TextToSpeech | None = None
        self.llm: LocalLLM | None = None
        self.knowledge_base: KnowledgeBase | None = None

        self._shutdown = False
        self._caffeinate: subprocess.Popen | None = None

    def run(self):
        """Main loop."""
        self._setup_signals()

        console.print("\n[bold green]╔══════════════════════════════════════════╗[/]")
        console.print("[bold green]║  COSIMO — Museum Docent (FREE Local)     ║[/]")
        console.print("[bold green]╚══════════════════════════════════════════╝[/]\n")

        # Check Ollama is running
        if not self._check_ollama():
            return

        # Prevent macOS sleep
        try:
            self._caffeinate = subprocess.Popen(
                ["caffeinate", "-dimsu"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

        console.print(f"  LLM Model: {self.ollama_model}")
        console.print(f"  TTS Voice: {self.tts_voice}")
        console.print(f"  Wake word: {self.wake_word.replace('_', ' ')}")
        console.print()

        # Initialize components
        self.knowledge_base = KnowledgeBase(self.collection_path)
        self.knowledge_base.load()

        self.wake_detector = WakeWordDetector(
            wake_word=self.wake_word,
            model_path=self.wake_word_model if self.wake_word_model else None,
        )
        self.stt = SpeechToText()
        self.tts = TextToSpeech(voice=self.tts_voice)
        self.llm = LocalLLM(model=self.ollama_model, knowledge_base=self.knowledge_base)

        while not self._shutdown:
            try:
                # IDLE state — listen for wake word
                console.print("[dim]Listening for wake word...[/]  Say [bold]'Hey Cosimo'[/]")
                detected = self.wake_detector.listen()

                if not detected or self._shutdown:
                    break

                console.print("[bold green]✦ Wake word detected![/]\n")

                # ACTIVE state — conversation loop
                self._run_conversation()

                console.print("[dim]Session ended — returning to wake word mode[/]\n")
                time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                import traceback
                traceback.print_exc()
                time.sleep(3)

        self._cleanup()
        console.print("\n[dim]Cosimo has stopped. Goodbye![/]")

    def _check_ollama(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            ollama_client = get_ollama()
            models = ollama_client.list()

            # Handle different response formats
            model_list = models.get('models', []) if isinstance(models, dict) else getattr(models, 'models', [])
            model_names = []
            for m in model_list:
                if isinstance(m, dict):
                    name = m.get('name', m.get('model', ''))
                else:
                    name = getattr(m, 'name', getattr(m, 'model', ''))
                if name:
                    model_names.append(name.split(':')[0])

            if self.ollama_model.split(':')[0] not in model_names:
                console.print(f"[yellow]Model '{self.ollama_model}' not found. Pulling...[/]")
                console.print("[dim]This may take a few minutes on first run.[/]")
                ollama_client.pull(self.ollama_model)

            return True
        except Exception as e:
            console.print(f"[red]Ollama not running or not installed![/]")
            console.print(f"[red]Error: {e}[/]")
            console.print("\n[yellow]To fix:[/]")
            console.print("  1. Install Ollama: https://ollama.ai")
            console.print("  2. Start Ollama: ollama serve")
            console.print(f"  3. Pull a model: ollama pull {self.ollama_model}")
            return False

    def _run_conversation(self):
        """Run a single conversation session."""
        self.llm.reset()

        # Greeting (short for faster start)
        greeting = "Hello! I'm Cosimo. What would you like to know about?"
        self.tts.speak(greeting)

        silence_count = 0
        max_silence = 3  # End after 3 consecutive silences

        while not self._shutdown:
            # Listen for user speech
            text = self.stt.listen_and_transcribe(timeout=15.0)

            if text is None:
                silence_count += 1
                if silence_count >= max_silence:
                    self.tts.speak("It was lovely chatting with you. Feel free to call on me again!")
                    break
                continue

            silence_count = 0

            # Get LLM response
            response, should_end = self.llm.chat(text)

            # Speak response
            self.tts.speak(response)

            if should_end:
                break

    def _setup_signals(self):
        def handler(sig, frame):
            self._shutdown = True
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _cleanup(self):
        if self.wake_detector:
            self.wake_detector.cleanup()
        if self.tts:
            self.tts.cleanup()
        if self._caffeinate:
            self._caffeinate.terminate()


def main():
    cosimo = CosimoLocal()
    cosimo.run()


if __name__ == "__main__":
    main()
