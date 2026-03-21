"""Test audio devices for the Cosimo kiosk."""

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    console.print("[bold]Cosimo Audio Test[/]\n")

    devices = sd.query_devices()
    table = Table(title="Audio Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Default", style="green")

    di, do = sd.default.device
    for i, d in enumerate(devices):
        flags = []
        if i == di: flags.append("IN")
        if i == do: flags.append("OUT")
        table.add_row(str(i), d["name"], str(d["max_input_channels"]),
                       str(d["max_output_channels"]), " ".join(flags))
    console.print(table)

    console.print("\n[bold]Mic test[/] — speak for 3 seconds")
    try:
        rec = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        peak = np.abs(rec).max()
        console.print(f"  Peak: {peak:.4f} ({20*np.log10(peak+1e-10):.1f} dBFS)")
        if peak < 0.01:
            console.print("[yellow]  ⚠ Very low — check mic[/]")
        elif peak > 0.95:
            console.print("[yellow]  ⚠ Clipping — reduce gain[/]")
        else:
            console.print("[green]  ✓ Levels OK[/]")
    except Exception as e:
        console.print(f"[red]  ✗ {e}[/]")

    console.print("\n[bold]Speaker test[/] — short tone")
    try:
        t = np.linspace(0, 0.5, 12000, endpoint=False)
        env = np.ones_like(t)
        env[:600] = np.linspace(0, 1, 600)
        env[-600:] = np.linspace(1, 0, 600)
        sd.play((0.3 * np.sin(2 * np.pi * 440 * t) * env).astype(np.float32), 24000)
        sd.wait()
        console.print("[green]  ✓ Done[/]")
    except Exception as e:
        console.print(f"[red]  ✗ {e}[/]")


if __name__ == "__main__":
    main()
