"""Verify exact bundled artificial fixtures across LF/CRLF checkout conversions.

Never relax content verification to filename/folder name. The hash in evidence
reports remains SHA256 of the original raw file, not of the normalized copy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Collection


MAX_BUNDLED_FIXTURE_BYTES = 64 * 1024


def matches_known_fixture(path: Path, raw_sha256: str, trusted_lf_hashes: Collection[str]) -> bool:
    """Accept trusted bytes or their CRLF-only checkout representation.

    Equivalent CRLF is accepted only if every newline is CRLF and converting
    back to LF gives a pre-existing, independently pinned SHA-256. Reject
    mixed-ending, edited, oversized, and arbitrary pseudo-synthetic files.
    """
    if raw_sha256 in trusted_lf_hashes:
        return True
    if path.stat().st_size > MAX_BUNDLED_FIXTURE_BYTES:
        return False
    data = path.read_bytes()
    if not data or b"\r\n" not in data:
        return False
    if data.count(b"\n") != data.count(b"\r\n"):
        return False
    canonical_sha = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    return canonical_sha in trusted_lf_hashes
