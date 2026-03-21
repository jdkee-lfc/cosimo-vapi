"""One-time setup: upload collection, create knowledge base, create Cosimo assistant on Vapi.

Usage:
    cosimo-setup                          # uses defaults
    cosimo-setup --collection data/collection.json
    cosimo-setup --reset                  # delete and recreate everything

This script:
  1. Uploads your museum collection JSON to Vapi's file storage
  2. Creates a Query Tool backed by that file (knowledge base)
  3. Creates (or updates) the Cosimo assistant with persona + tool
  4. Writes the VAPI_ASSISTANT_ID to your .env file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv, set_key
from rich.console import Console

from cosimo_vapi.persona import FIRST_MESSAGE, SYSTEM_PROMPT

console = Console()

VAPI_BASE = "https://api.vapi.ai"


def api(method: str, path: str, token: str, **kwargs) -> dict:
    """Make an authenticated request to Vapi API."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # For file uploads, Content-Type is set by httpx
    if "files" in kwargs:
        headers.pop("Content-Type", None)
    resp = httpx.request(method, f"{VAPI_BASE}{path}", headers=headers, timeout=60, **kwargs)
    if resp.status_code >= 400:
        console.print(f"[red]API error {resp.status_code}:[/] {resp.text}")
        sys.exit(1)
    return resp.json() if resp.text else {}


def upload_file(token: str, filepath: Path) -> str:
    """Upload a file to Vapi and return its file ID."""
    console.print(f"  Uploading {filepath.name} ({filepath.stat().st_size / 1024:.0f} KB)...")
    with open(filepath, "rb") as f:
        resp = httpx.post(
            f"{VAPI_BASE}/file",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (filepath.name, f, "application/json")},
            timeout=120,
        )
    if resp.status_code >= 400:
        console.print(f"[red]Upload failed ({resp.status_code}):[/] {resp.text}")
        sys.exit(1)
    data = resp.json()
    file_id = data["id"]
    console.print(f"  [green]✓[/] File uploaded: {file_id}")
    return file_id


def create_query_tool(token: str, file_id: str) -> str:
    """Create a Query Tool with the uploaded file as knowledge base."""
    console.print("  Creating knowledge base query tool...")
    payload = {
        "type": "query",
        "function": {
            "name": "knowledge-search",
            "description": (
                "Search the museum collection database. Use this tool whenever a visitor "
                "asks about any artwork, artist, gallery, period, medium, or anything "
                "related to items in the museum. Always search before answering questions "
                "about specific artworks or the collection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query about artworks, artists, or collection items.",
                    }
                },
                "required": ["query"],
            },
        },
        "knowledgeBases": [
            {
                "provider": "google",
                "fileIds": [file_id],
                "name": "Museum Collection",
                "description": (
                    "Complete catalog of the museum's collection including artworks, "
                    "artists, dates, mediums, dimensions, descriptions, gallery locations, "
                    "periods, provenance, and related items."
                ),
            }
        ],
    }
    data = api("POST", "/tool", token, json=payload)
    tool_id = data["id"]
    console.print(f"  [green]✓[/] Query tool created: {tool_id}")
    return tool_id


def create_assistant(token: str, tool_id: str) -> str:
    """Create the Cosimo voice assistant with persona and knowledge tool."""
    console.print("  Creating Cosimo assistant...")
    payload = {
        "name": "Cosimo — Museum Docent",
        "firstMessage": FIRST_MESSAGE,
        "model": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.4,
            "maxTokens": 300,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ],
            "toolIds": [tool_id],
        },
        "voice": {
            "provider": "11labs",
            "voiceId": "pNInz6obpgDQGcFmaJgB",  # "Adam" — warm, authoritative male voice
            "model": "eleven_flash_v2_5",
            "stability": 0.5,
            "similarityBoost": 0.75,
        },
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-3",
            "language": "en",
        },
        # Conversation behavior
        "silenceTimeoutSeconds": 20,
        "maxDurationSeconds": 600,  # 10-minute max per session
        "backgroundSound": "off",
        "backchannelingEnabled": True,
        # Interruption handling
        "clientMessages": [
            "conversation-update",
            "function-call",
            "hang",
            "model-output",
            "speech-update",
            "status-update",
            "transcript",
            "tool-calls",
            "user-interrupted",
            "voice-input",
        ],
        "serverMessages": [
            "conversation-update",
            "end-of-call-report",
            "hang",
            "speech-update",
            "status-update",
            "tool-calls",
        ],
    }
    data = api("POST", "/assistant", token, json=payload)
    assistant_id = data["id"]
    console.print(f"  [green]✓[/] Assistant created: {assistant_id}")
    return assistant_id


def update_assistant(token: str, assistant_id: str, tool_id: str) -> str:
    """Update an existing assistant with new tool and prompt."""
    console.print(f"  Updating assistant {assistant_id}...")
    payload = {
        "name": "Cosimo — Museum Docent",
        "firstMessage": FIRST_MESSAGE,
        "model": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.4,
            "maxTokens": 300,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ],
            "toolIds": [tool_id],
        },
        "voice": {
            "provider": "11labs",
            "voiceId": "pNInz6obpgDQGcFmaJgB",
            "model": "eleven_flash_v2_5",
            "stability": 0.5,
            "similarityBoost": 0.75,
        },
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-3",
            "language": "en",
        },
        "silenceTimeoutSeconds": 20,
        "maxDurationSeconds": 600,
        "backgroundSound": "off",
        "backchannelingEnabled": True,
    }
    data = api("PATCH", f"/assistant/{assistant_id}", token, json=payload)
    console.print(f"  [green]✓[/] Assistant updated")
    return data["id"]


def main():
    parser = argparse.ArgumentParser(description="Set up Cosimo on Vapi")
    parser.add_argument(
        "--collection", default="data/collection.json",
        help="Path to museum collection JSON (default: data/collection.json)",
    )
    parser.add_argument("--reset", action="store_true", help="Create a new assistant even if one exists")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("VAPI_API_KEY", "")
    if not token:
        console.print("[red]✗[/] VAPI_API_KEY not set in .env")
        console.print("  Get your API key at https://dashboard.vapi.ai/")
        sys.exit(1)

    collection_path = Path(args.collection)
    if not collection_path.exists():
        console.print(f"[red]✗[/] Collection not found: {collection_path}")
        console.print("  Place your museum collection JSON there, or use --collection PATH")
        console.print("  See data/sample_collection.json for the expected format.")
        sys.exit(1)

    # Validate JSON
    with open(collection_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("items", data.get("collection", data.get("objects", [])))
    elif isinstance(data, list):
        items = data
    else:
        items = []
    console.print(f"\n[bold]Cosimo — Vapi Setup[/]\n")
    console.print(f"  Collection: {collection_path} ({len(items)} items, {collection_path.stat().st_size / 1024:.0f} KB)")

    # Check file size — Vapi recommends < 300KB per file
    file_size_kb = collection_path.stat().st_size / 1024
    if file_size_kb > 300:
        console.print(f"  [yellow]⚠ File is {file_size_kb:.0f} KB (Vapi recommends < 300KB for best retrieval)[/]")
        console.print(f"  [yellow]  Consider splitting into multiple files if retrieval quality is low.[/]")

    console.print()

    # Step 1: Upload file
    file_id = upload_file(token, collection_path)

    # Step 2: Create query tool / knowledge base
    tool_id = create_query_tool(token, file_id)

    # Step 3: Create or update assistant
    existing_id = os.getenv("VAPI_ASSISTANT_ID", "").strip()
    if existing_id and not args.reset:
        assistant_id = update_assistant(token, existing_id, tool_id)
    else:
        assistant_id = create_assistant(token, tool_id)

    # Step 4: Save assistant ID to .env
    env_path = Path(".env")
    if env_path.exists():
        set_key(str(env_path), "VAPI_ASSISTANT_ID", assistant_id)
        console.print(f"\n  [green]✓[/] Saved VAPI_ASSISTANT_ID={assistant_id} to .env")
    else:
        console.print(f"\n  [yellow]⚠[/] No .env file found — add this to your .env:")
        console.print(f"    VAPI_ASSISTANT_ID={assistant_id}")

    console.print(f"\n[bold green]Setup complete![/]")
    console.print(f"  Run [bold]cosimo[/] to start the docent.\n")


if __name__ == "__main__":
    main()
