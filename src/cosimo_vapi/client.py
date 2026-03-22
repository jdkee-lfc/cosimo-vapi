"""Cosimo local client — wake word detection → Vapi voice call → repeat.

This is the thin local layer that runs on the Mac Studio kiosk.
All speech recognition, LLM reasoning, and text-to-speech happen
in the cloud via Vapi. Locally we only handle:

  1. Always-on wake word detection ("Cosimo") via Porcupine
  2. Starting/stopping Vapi web calls via the Python client SDK
  3. Monitoring call state for session lifecycle

State machine:
  IDLE → [wake word detected] → ACTIVE (Vapi call) → [call ends] → IDLE
"""

from __future__ import annotations

import asyncio
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pvporcupine
import sounddevice as sd
from dotenv import load_dotenv
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Wake word detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Listens continuously for 'Cosimo' using Picovoice Porcupine."""

    def __init__(self, access_key: str, keyword_path: str | None = None, sensitivity: float = 0.6):
        self.access_key = access_key
        self.keyword_path = keyword_path
        self.sensitivity = sensitivity
        self._porcupine: pvporcupine.Porcupine | None = None

    def _init(self):
        if self._porcupine is not None:
            return

        kw_path = self.keyword_path
        if kw_path and Path(kw_path).exists():
            console.print(f"  Wake word model: {kw_path}")
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[kw_path],
                sensitivities=[self.sensitivity],
            )
        else:
            if kw_path:
                console.print(f"  [yellow]⚠ Custom model not found: {kw_path}[/]")
            console.print("  [yellow]Using built-in 'computer' keyword for testing[/]")
            console.print("  [dim]Generate 'Cosimo' at https://console.picovoice.ai/[/]")
            self._porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=["computer"],
                sensitivities=[self.sensitivity],
            )

    def listen(self) -> bool:
        """Block until wake word is detected. Returns True on detection."""
        self._init()
        porcupine = self._porcupine
        frame_length = porcupine.frame_length

        buffer = b""
        bytes_per_frame = frame_length * 2  # int16

        detected = False

        def callback(indata, frames, time_info, status):
            nonlocal buffer, detected
            if status:
                pass  # ignore overflows silently in kiosk mode
            pcm = (indata[:, 0] * 32767).astype(np.int16)
            buffer += pcm.tobytes()

            while len(buffer) >= bytes_per_frame:
                frame_bytes = buffer[:bytes_per_frame]
                buffer = buffer[bytes_per_frame:]
                pcm_frame = struct.unpack_from(f"{frame_length}h", frame_bytes)
                if porcupine.process(pcm_frame) >= 0:
                    detected = True

        stream = sd.InputStream(
            samplerate=porcupine.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_length,
            callback=callback,
        )

        with stream:
            sd.sleep(500)  # Discard 500ms of stale audio
            detected = False  # Reset in case stale audio triggered it
            buffer = b""     # Clear buffer
            while not detected:
                sd.sleep(100)

        return True

    def cleanup(self):
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None


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
        self.pv_key = os.getenv("PICOVOICE_ACCESS_KEY", "")

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
        console.print(f"  Wake word: Cosimo")
        console.print()

        self.wake_detector = WakeWordDetector(
            access_key=self.pv_key,
            keyword_path=os.getenv("WAKE_WORD_PATH", "data/cosimo_wake_word.ppn"),
        )

        while not self._shutdown:
            try:
                # IDLE state — listen for wake word
                console.print("[dim]Listening for wake word...[/]  Say [bold]'Cosimo'[/]")
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
        if not self.pv_key:
            errors.append("PICOVOICE_ACCESS_KEY not set — get it at https://console.picovoice.ai/")

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
