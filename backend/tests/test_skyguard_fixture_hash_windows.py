"""Security regression: LF/CRLF are the only accepted checkout variants.

All tests are software-only; no camera or inference used.
"""
import hashlib
from pathlib import Path

import pytest

from app.evaluation.fixture_hash_v1 import matches_known_fixture

FIXTURES = (
    ("phase42_4_SYNTHETIC_timing.jsonl", "d6ea5fb39426dd237900245a943edd5f326e92d441dad49836142a925eb7b892"),
    ("phase42_1_synthetic_crossing.jsonl", "a6f84c5853b4c2f0f51ed3fe9e44abd39246b33c38221ba5010210758ab7e291"),
    ("phase42_3_SYNTHETIC_identity_swap.jsonl", "3ab4cb2be46223ca6c82579f9989562e630eda2a4bc12d3553babb5325076a34"),
)


@pytest.mark.parametrize("filename,trusted_hash", FIXTURES)
def test_lf_and_crlf_exact_content_only(tmp_path, filename, trusted_hash):
    original = Path(__file__).resolve().parents[1] / "examples" / filename
    canonical = original.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == trusted_hash
    for endings in (canonical, canonical.replace(b"\n", b"\r\n")):
        probe = tmp_path / filename
        probe.write_bytes(endings)
        raw = hashlib.sha256(endings).hexdigest()
        assert matches_known_fixture(probe, raw, frozenset({trusted_hash}))
    tampered = canonical.replace(b"0", b"1", 1)
    assert tampered != canonical
    probe.write_bytes(tampered)
    assert not matches_known_fixture(probe, hashlib.sha256(tampered).hexdigest(), frozenset({trusted_hash}))


def test_reject_mixed_newline_in_fixture(tmp_path):
    raw = b"{\"frame\":1}\r\n{\"frame\":2}\n"
    probe = tmp_path / "mixed.jsonl"
    probe.write_bytes(raw)
    assert not matches_known_fixture(probe, hashlib.sha256(raw).hexdigest(), {"a"*64})


def test_reject_giant_untrusted_fixture(tmp_path):
    raw = b"hello\r\n" * 10000
    probe = tmp_path / "large.jsonl"
    probe.write_bytes(raw)
    assert not matches_known_fixture(probe, hashlib.sha256(raw).hexdigest(), {"a"*64})
