"""Download the Inconel-625 turning dataset (Mendeley Data v3, CC BY 4.0) and verify sha256.

Only the 'Synchronized' folder is used: force and acceleration are already trimmed to the
stable cutting window and aligned sample by sample (see the dataset's Data_Sync_Algorithm.pdf).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from mrl.config import MANIFEST, RAW


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    entries = json.loads(MANIFEST.read_text())
    bad = 0
    for e in entries:
        out = RAW / e["folder"] / e["filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if not (out.exists() and out.stat().st_size == e["size"]):
            subprocess.run(["curl", "-sL", "--retry", "3", e["url"], "-o", str(out)], check=True)
        ok = sha256(out) == e["sha256"]
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {e['folder']}/{e['filename']}")
    print(f"{len(entries) - bad}/{len(entries)} files verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
