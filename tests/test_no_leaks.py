"""The repo must not carry identity, credentials, or one machine's paths.

This started as a one-off scan before the first publish. It is a test because
the things it catches come back: a debug script with an absolute path, a key
pasted into a docstring, an internal endpoint in a config example.

The `internal infra` pattern grew when the training and corpus code was lifted
in from the cluster it ran on. That code was written against a job scheduler, a
shared filesystem and an internal package mirror, and each of those leaves a
recognisable word behind. A reader outside the lab cannot use any of them, so
none of them belongs here -- the generic name for the thing does.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '__pycache__', 'runs', 'scores', 'node_modules', '.pytest_cache'}
# .gif was missing from the original list, and a GIF's compressed bytes
# contain enough ASCII to match a short pattern three times over.
BINARY = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.gz', '.zip', '.pck',
          '.wasm', '.ttf', '.otf', '.woff', '.woff2', '.ico', '.pdf', '.bin', '.so'}

PATTERNS = {
    'credential': re.compile(
        r'sk-(?!x{4})[A-Za-z0-9_\-]{20,}|hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_\-]{30,}'
        r'|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}'
        r'|-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    # Any user's home directory, not a list of particular ones.
    'machine path': re.compile(r'/home/[A-Za-z_][A-Za-z0-9_.\-]*/|/Users/[A-Za-z_][A-Za-z0-9_.\-]*/'
                               r'|/storage/[A-Za-z_][A-Za-z0-9_.\-]*/'),
    'personal email': re.compile(r'[A-Za-z0-9._%+\-]+@(?!example\.)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'),
}
# The scan reads itself, so its own patterns would match. Nothing else is exempt.
EXEMPT = {'tests/test_no_leaks.py'}


def files():
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() in BINARY:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = str(p.relative_to(ROOT))
        if rel in EXEMPT:
            continue
        yield p, rel


@pytest.mark.parametrize('kind', sorted(PATTERNS))
def test_no_leaks(kind):
    rx = PATTERNS[kind]
    hits = []
    for p, rel in files():
        try:
            text = p.read_text(errors='replace')
        except OSError:
            continue
        for m in rx.finditer(text):
            line = text[:m.start()].count('\n') + 1
            hits.append(f'{rel}:{line}: {m.group(0)[:60]}')
    assert not hits, f'{kind} found:\n  ' + '\n  '.join(hits[:20])
