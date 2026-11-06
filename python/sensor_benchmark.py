#!/usr/bin/env python3
"""Subject-disjoint UCI HAR evaluation. Training and validation are C++; official test is scored once."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

SOURCE_URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
EXPECTED_SHA256 = None  # Pinned after the initial official-source acquisition.
VALIDATION_SUBJECTS = (3, 15, 17, 26, 30)
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024

def acquire(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "uci-har-240.zip"
    if not path.exists():
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=90) as response, temporary.open("wb") as out:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("Dataset archive exceeds download limit")
                    out.write(chunk)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Dataset archive exceeds size limit")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if EXPECTED_SHA256 is not None and digest != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch; cached data must not be used")
    print(json.dumps({"source": SOURCE_URL, "archive_sha256": digest,
                      "pinned": EXPECTED_SHA256 is not None}), flush=True)
    return path, digest

def selected_zip(path):
    outer = zipfile.ZipFile(path)
    nested = [n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")]
    if nested:
        entry = outer.getinfo(nested[0])
        if entry.file_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Nested archive exceeds size limit")
        inner = zipfile.ZipFile(io.BytesIO(outer.read(entry)))
        outer.close()
        return inner
    return outer

def member(archive, suffix):
    matches = [n for n in archive.namelist() if n.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one dataset member: {suffix}")
    info = archive.getinfo(matches[0])
    if info.file_size > 100 * 1024 * 1024:
        raise ValueError("Dataset member exceeds size limit")
    return archive.read(info).decode("utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/uci-har"))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    path, digest = acquire(args.cache)
    if not args.download_only:
        with selected_zip(path) as archive:
            print(json.dumps({"feature_file": member(archive, "features.txt").splitlines()[:3],
                              "archive_sha256": digest}), flush=True)
if __name__ == "__main__":
    main()
