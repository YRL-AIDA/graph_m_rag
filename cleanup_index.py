#!/usr/bin/env python3
"""
Delete from the KiloCode index (ws-22557bd7a1ab437d) all points belonging to
files that are NOT source code. Keeps only:
  - *.py (Python source)
  - *.md (project documentation / specs)
  - app/static/ask-document.html (frontend source)
Everything else (generated test results JSON/HTML, data files, unknown) is removed.
"""
import json
import urllib.request
from collections import Counter

QD = "http://0.0.0.0:16333"
COLLECTION = "ws-22557bd7a1ab437d"


def api(path, body):
    req = urllib.request.Request(
        f"{QD}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def scroll_all():
    """Return {filePath: [point_ids]} for all points."""
    files = {}
    unknown_ids = []
    offset = None
    while True:
        body = {"limit": 5000, "with_payload": ["filePath"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        data = api(f"/collections/{COLLECTION}/points/scroll", body)
        pts = data.get("result", {}).get("points", [])
        if not pts:
            break
        for p in pts:
            fp = p.get("payload", {}).get("filePath")
            if fp is None or fp == "<unknown>":
                unknown_ids.append(p["id"])
                continue
            files.setdefault(fp, []).append(p["id"])
        offset = data.get("result", {}).get("next_page_offset")
        if not offset:
            break
    return files, unknown_ids


def is_source_code(fp: str) -> bool:
    if fp == "app/static/ask-document.html":
        return True
    base = fp.rsplit("/", 1)[-1]
    if "." not in base:
        return False
    return base.rsplit(".", 1)[-1].lower() in {"py", "md"}


def delete_points(ids):
    # delete in batches of 5000
    for i in range(0, len(ids), 5000):
        batch = ids[i:i + 5000]
        api(f"/collections/{COLLECTION}/points/delete", {"points": batch})
        print(f"  deleted batch {i}-{i + len(batch)} ({len(batch)} points)")


def main():
    files, unknown_ids = scroll_all()
    total_points = sum(len(v) for v in files.values()) + len(unknown_ids)
    print(f"Total files: {len(files)}, unknown points: {len(unknown_ids)}, total points: {total_points}")

    keep = {}
    delete = {}
    for fp, ids in files.items():
        if is_source_code(fp):
            keep[fp] = ids
        else:
            delete[fp] = ids

    keep_points = sum(len(v) for v in keep.values())
    del_points = sum(len(v) for v in delete.values()) + len(unknown_ids)
    print(f"KEEP:   {len(keep)} files / {keep_points} points")
    print(f"DELETE: {len(delete)} files / {del_points} points")

    print("\n=== FILES TO DELETE ===")
    for fp, ids in sorted(delete.items()):
        print(f"  {len(ids):6d}  {fp}")
    if unknown_ids:
        print(f"  {len(unknown_ids):6d}  <unknown>")

    print("\n=== FILES KEPT ===")
    for fp, ids in sorted(keep.items()):
        print(f"  {len(ids):6d}  {fp}")

    # Collect all ids to delete
    all_delete_ids = list(unknown_ids)
    for ids in delete.values():
        all_delete_ids.extend(ids)
    print(f"\nDeleting {len(all_delete_ids)} points...")
    delete_points(all_delete_ids)
    print("Done.")


if __name__ == "__main__":
    main()
