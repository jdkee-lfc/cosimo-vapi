"""Cosimo local client — wake word detection → Vapi voice call → repeat.

This is the thin local layer that runs on the Mac Studio kiosk.
All speech recognition, LLM reasoning, and text-to-speech happen
in the cloud via Vapi. Locally we only handle:

  1. Always-on wake word detection via OpenWakeWord (open source, no account needed)
  2. Starting/stopping Vapi web calls via the Python client SDK
  3. Monitoring call state for session lifecycle

State machine:
  IDLE → [wake word detected] → ACTIVE (Vapi call) → [call ends] → IDLE
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openwakeword
from openwakeword.model import Model as OWWModel
import sounddevice as sd
import torch
import torchaudio
from dotenv import load_dotenv
from rich.console import Console

console = Console()

# Default wake word - can be changed in .env via WAKE_WORD setting
DEFAULT_WAKE_WORD = "hey_cosimo"
SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # 80ms at 16kHz


# ---------------------------------------------------------------------------
# Custom wake word model (trained locally)
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
# Wake word detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Listens continuously for wake word using custom model or OpenWakeWord."""

    def __init__(self, wake_word: str = DEFAULT_WAKE_WORD, model_path: str | None = None, threshold: float = 0.5):
        self.wake_word = wake_word
        self.model_path = model_path  # Path to custom .pt or .onnx model
        self.threshold = threshold
        self._model = None
        self._custom_model = None  # For custom PyTorch model
        self._use_custom = False
        self._initialized = False
        self._audio_buffer = []

    def _init(self):
        if self._initialized:
            return

        console.print("  [dim]Loading wake word models...[/]")

        # Check for custom PyTorch model (.pt file)
        if self.model_path and self.model_path.endswith('.pt') and Path(self.model_path).exists():
            console.print(f"  [dim]Using custom PyTorch model: {self.model_path}[/]")
            self._custom_model = CustomWakeWordModel()
            self._custom_model.load_state_dict(
                torch.load(self.model_path, weights_only=True, map_location='cpu')
            )
            self._custom_model.eval()
            self._use_custom = True
        # Check for custom ONNX model
        elif self.model_path and Path(self.model_path).exists():
            console.print(f"  [dim]Using custom model: {self.model_path}[/]")
            self._model = OWWModel(
                wakeword_models=[self.model_path],
                inference_framework="onnx"
            )
        else:
            # Download and use pre-trained models
            openwakeword.utils.download_models()
            self._model = OWWModel(inference_framework="onnx")

        self._initialized = True
        console.print(f"  Wake word: [bold]{self.wake_word.replace('_', ' ')}[/]")

    def listen(self) -> bool:
        """Block until wake word is detected. Returns True on detection."""
        self._init()
        detected = False
        self._audio_buffer = []

        if self._use_custom:
            # Use custom PyTorch model
            def callback(indata, frames, time_info, status):
                nonlocal detected
                if status or detected:
                    return

                # Accumulate audio (need 1 second = 16000 samples)
                self._audio_buffer.extend(indata[:, 0].tolist())

                # Keep only last 1 second
                if len(self._audio_buffer) > 16000:
                    self._audio_buffer = self._audio_buffer[-16000:]

                # Only predict when we have enough audio
                if len(self._audio_buffer) >= 16000:
                    audio = torch.FloatTensor(self._audio_buffer[-16000:]).unsqueeze(0)
                    with torch.no_grad():
                        prob = self._custom_model(audio).item()
                    if prob > self.threshold:
                        detected = True
                        return
        else:
            # Use OpenWakeWord model
            model = self._model

            def callback(indata, frames, time_info, status):
                nonlocal detected
                if status or detected:
                    return

                pcm = (indata[:, 0] * 32767).astype(np.int16)
                predictions = model.predict(pcm)

                for name, score in predictions.items():
                    if self.wake_word in name.lower() and score > self.threshold:
                        detected = True
                        return

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            callback=callback,
        )

        with stream:
            sd.sleep(500)  # Discard 500ms of stale audio
            detected = False  # Reset
            self._audio_buffer = []
            if not self._use_custom and self._model:
                self._model.reset()
            while not detected:
                sd.sleep(100)

        return True

    def cleanup(self):
        self._model = None
        self._custom_model = None
        self._initialized = False


# ---------------------------------------------------------------------------
# Vapi call manager
# ---------------------------------------------------------------------------

class VapiCallManager:
    """Runs each Vapi call in a subprocess for clean Daily context."""

    def __init__(self, public_key: str, assistant_id: str):
        self.public_key = public_key
        self.assistant_id = assistant_id
        self._proc = None

    def start_call(self):
        """Launch the call in a subprocess."""
        import subprocess as sp
        worker = Path(__file__).parent / "call_worker.py"
        self._proc = sp.Popen(
            [sys.executable, str(worker), self.public_key, self.assistant_id],
        )
        console.print("[bold blue]Starting Vapi call...[/]")
        time.sleep(2)  # Give it time to connect
        console.print("[green]✦ Call connected — Cosimo is speaking[/]")

    def wait_for_end(self):
        """Block until the subprocess dies."""
        console.print("\n[dim]Conversation active...[/]")
        if self._proc:
            self._proc.wait()
        console.print("[dim]Session ending...[/]")

    def stop(self):
        """Kill the subprocess."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        console.print("[dim]Call ended[/]")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class Cosimo:
    """Top-level controller: wake word → Vapi call → repeat."""

    def __init__(self):
        load_dotenv()

        self.public_key = os.getenv("VAPI_PUBLIC_KEY", "")
        self.assistant_id = os.getenv("VAPI_ASSISTANT_ID", "")
        self.wake_word = os.getenv("WAKE_WORD", DEFAULT_WAKE_WORD)
        self.wake_word_model = os.getenv("WAKE_WORD_MODEL", "")

        self.wake_detector: WakeWordDetector | None = None
        self.call_manager: VapiCallManager | None = None
        self._shutdown = False
        self._caffeinate: subprocess.Popen | None = None

    def run(self):
        """Main loop."""
        self._validate()
        self._setup_signals()

        console.print("\n[bold green]╔══════════════════════════════════════╗[/]")
        console.print("[bold green]║  COSIMO — Museum Docent (Vapi Cloud) ║[/]")
        console.print("[bold green]╚══════════════════════════════════════╝[/]\n")

        # Prevent macOS sleep
        try:
            self._caffeinate = subprocess.Popen(
                ["caffeinate", "-dimsu"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

        console.print(f"  Assistant: {self.assistant_id[:20]}...")
        console.print(f"  Wake word: {self.wake_word.replace('_', ' ')}")
        console.print()

        self.wake_detector = WakeWordDetector(
            wake_word=self.wake_word,
            model_path=self.wake_word_model if self.wake_word_model else None,
        )

        while not self._shutdown:
            try:
                # IDLE state — listen for wake word
                console.print("[dim]Listening for wake word...[/]  Say [bold]'Hey Cosimo'[/]")
                detected = self.wake_detector.listen()

                if not detected or self._shutdown:
                    break

                console.print("[bold green]✦ Wake word detected![/]\n")

                # ACTIVE state — run Vapi call
                self.call_manager = VapiCallManager(self.public_key, self.assistant_id)
                try:
                    self.call_manager.start_call()
                    self.call_manager.wait_for_end()
                finally:
                    self.call_manager.stop()
                    self.call_manager = None
                    time.sleep(2)  # Let mic fully release from subprocess

                console.print("[dim]Session ended — returning to wake word mode[/]\n")
                time.sleep(1)  # Brief pause between sessions

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                time.sleep(3)

        self._cleanup()
        console.print("\n[dim]Cosimo has stopped. Goodbye![/]")

    def _validate(self):
        """Check required configuration."""
        errors = []
        if not self.public_key:
            errors.append("VAPI_PUBLIC_KEY not set — get it at https://dashboard.vapi.ai/")
        if not self.assistant_id:
            errors.append("VAPI_ASSISTANT_ID not set — run cosimo-setup first")

        if errors:
            console.print("[bold red]Missing configuration:[/]\n")
            for e in errors:
                console.print(f"  [red]✗[/] {e}")
            console.print("\n  Edit your .env file to add these values.")
            sys.exit(1)

    def _setup_signals(self):
        def handler(sig, frame):
            self._shutdown = True
            if self.call_manager:
                self.call_manager.stop()
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _cleanup(self):
        if self.wake_detector:
            self.wake_detector.cleanup()
        if self.call_manager:
            self.call_manager.stop()
        if self._caffeinate:
            self._caffeinate.terminate()


def main():
    cosimo = Cosimo()
    cosimo.run()


if __name__ == "__main__":
    main()
