#!/usr/bin/env python3
"""Enumerate all distinct file paths indexed in the KiloCode workspace Qdrant collection."""
import json
import urllib.request
from collections import Counter, defaultdict

QD = "http://0.0.0.0:16333"
COLLECTION = "ws-22557bd7a1ab437d"


def scroll_all():
    paths = Counter()
    per_ext = defaultdict(Counter)
    offset = None
    total = 0
    while True:
        body = {
            "limit": 5000,
            "with_payload": ["filePath"],
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{QD}/collections/{COLLECTION}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        pts = data.get("result", {}).get("points", [])
        if not pts:
            break
        total += len(pts)
        for p in pts:
            fp = p.get("payload", {}).get("filePath", "<unknown>")
            paths[fp] += 1
            ext = fp.rsplit(".", 1)[-1].lower() if "." in fp.rsplit("/", 1)[-1] else "<noext>"
            per_ext[ext][fp] += 1
        offset = data.get("result", {}).get("next_page_offset")
        if not offset:
            break
    return total, paths, per_ext


def main():
    total, paths, per_ext = scroll_all()
    print(f"TOTAL POINTS SCROLLED: {total}")
    print(f"DISTINCT FILES: {len(paths)}")
    print("\n=== BY EXTENSION ===")
    for ext in sorted(per_ext, key=lambda e: -sum(per_ext[e].values())):
        files = per_ext[ext]
        n = sum(files.values())
        print(f"{ext:12s} points={n:6d} files={len(files):4d}")
    print("\n=== ALL FILES (path: points) ===")
    for fp, cnt in sorted(paths.items()):
        print(f"{cnt:6d}  {fp}")


if __name__ == "__main__":
    main()
