#!/usr/bin/env python3
"""Split and optimize collection.json for better Vapi RAG performance.

This script:
1. Removes unnecessary fields (sourcePath, sourceFile, searchText, etc.)
2. Splits the collection by category into smaller files
3. Preserves the original collection.json

Usage:
    python scripts/split_collection.py
    python scripts/split_collection.py --input data/collection.json --output-dir data/
"""

import argparse
import json
from pathlib import Path


# Fields to remove (administrative/redundant - not needed for RAG)
FIELDS_TO_REMOVE = [
    "sourcePath",      # Windows file path - not useful
    "sourceFile",      # Internal reference
    "searchText",      # Redundant with description
    "catalogedBy",     # Administrative
    "catalogDate",     # Administrative
    "status",          # Administrative
    "makersMark",      # Usually empty
]

# Category groupings for splitting
CATEGORY_GROUPS = {
    "paintings_galleries": ["Painting"],  # Will be filtered by room
    "paintings_rooms": ["Painting"],      # Will be filtered by room
    "sculptures": ["Sculpture"],
    "furniture_other": ["Furniture", "Other", "Textile", "Decorative"],
}

# Rooms that are galleries (for splitting paintings)
GALLERY_ROOMS = [
    "Upstairs, East Gallery",
    "Downstairs, South Gallery",
    "Upstairs, South Gallery",
    "Upstairs, North Gallery",
    "Stair Hall/West Gallery",
    "Stair Wall/West Gallery",
    "Downstairs, South Galler",  # Typo in data
]


def clean_item(item: dict) -> dict:
    """Remove unnecessary fields and clean up empty nested objects."""
    cleaned = {k: v for k, v in item.items() if k not in FIELDS_TO_REMOVE}

    # Clean up empty inscription objects
    if "inscription" in cleaned:
        insc = cleaned["inscription"]
        if isinstance(insc, dict) and all(not v for v in insc.values()):
            del cleaned["inscription"]

    # Clean up dimensions - remove null values
    if "dimensions" in cleaned and isinstance(cleaned["dimensions"], dict):
        cleaned["dimensions"] = {k: v for k, v in cleaned["dimensions"].items() if v is not None and v != ""}

    return cleaned


def split_collection(input_path: Path, output_dir: Path) -> dict:
    """Split collection into category-based files."""

    with open(input_path) as f:
        data = json.load(f)

    items = data.get("items", [])
    print(f"Loaded {len(items)} items from {input_path}")

    # Clean all items
    cleaned_items = [clean_item(item) for item in items]

    # Group by category (with special handling for paintings)
    grouped = {
        "paintings_galleries": [],
        "paintings_rooms": [],
        "sculptures": [],
        "furniture_other": [],
    }

    for item in cleaned_items:
        category = item.get("category", "Other")
        room = item.get("room", "")

        if category == "Painting":
            # Split paintings by gallery vs house rooms
            if room in GALLERY_ROOMS:
                grouped["paintings_galleries"].append(item)
            else:
                grouped["paintings_rooms"].append(item)
        elif category == "Sculpture":
            grouped["sculptures"].append(item)
        else:
            grouped["furniture_other"].append(item)

    # Write split files
    results = {}
    for group_name, group_items in grouped.items():
        if not group_items:
            continue

        output_path = output_dir / f"collection_{group_name}.json"
        output_data = {"items": group_items}

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        size_kb = output_path.stat().st_size / 1024
        results[group_name] = {
            "path": output_path,
            "count": len(group_items),
            "size_kb": size_kb,
        }
        print(f"  {output_path.name}: {len(group_items)} items, {size_kb:.1f} KB")

    # Also create a single optimized file (all items, cleaned)
    optimized_path = output_dir / "collection_optimized.json"
    with open(optimized_path, "w") as f:
        json.dump({"items": cleaned_items}, f, indent=2)

    opt_size = optimized_path.stat().st_size / 1024
    print(f"  {optimized_path.name}: {len(cleaned_items)} items, {opt_size:.1f} KB (all items, optimized)")

    results["optimized"] = {
        "path": optimized_path,
        "count": len(cleaned_items),
        "size_kb": opt_size,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Split and optimize collection.json")
    parser.add_argument(
        "--input", "-i",
        default="data/collection.json",
        help="Input collection JSON file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/",
        help="Output directory for split files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSplitting collection for Vapi optimization...")
    print(f"  Input: {input_path} ({input_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Output: {output_dir}/")
    print()

    original_size = input_path.stat().st_size / 1024
    results = split_collection(input_path, output_dir)

    print()
    print("Summary:")
    print(f"  Original: {original_size:.1f} KB")
    total_split = sum(r["size_kb"] for name, r in results.items() if name != "optimized")
    print(f"  Split files total: {total_split:.1f} KB")
    print(f"  Savings: {original_size - results['optimized']['size_kb']:.1f} KB ({(1 - results['optimized']['size_kb']/original_size)*100:.1f}%)")
    print()
    print("Files created:")
    for name, info in results.items():
        status = "OK" if info["size_kb"] < 300 else "WARNING: >300KB"
        print(f"  {info['path'].name}: {info['count']} items, {info['size_kb']:.1f} KB [{status}]")

    return 0


if __name__ == "__main__":
    exit(main())
