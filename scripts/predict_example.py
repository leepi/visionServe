"""Tiny example client — hits the local API with one image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    with args.image.open("rb") as f:
        files = {"file": (args.image.name, f, "image/jpeg")}
        r = httpx.post(f"{args.url}/predict", params={"top_k": args.top_k}, files=files, timeout=30)

    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
