"""The repo must not carry identity, credentials, or one machine's paths.

This started as a one-off scan before the first publish. It is a test because
the things it catches come back: a debug script with an absolute path, a key
pasted into a docstring, an internal endpoint in a config example.

Names of people and internal infrastructure are not spelled out here, nor
hashed (short words hash back by dictionary): a list of them in the repo would
itself be the leak. They live in tests/leak_terms.local.txt, which is
gitignored; one term per line, # for comments. Without that file the
`internal names` check is skipped.
"""
from __future__ import annotations

import re
import subprocess
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

LOCAL_TERMS = ROOT / 'tests' / 'leak_terms.local.txt'
WORD = re.compile(r'[a-z0-9][a-z0-9.\-]*')

# The scan reads itself, so its own patterns would match. Nothing else is exempt.
EXEMPT = {'tests/test_no_leaks.py'}


def local_terms():
    if not LOCAL_TERMS.is_file():
        return set()
    lines = (l.split('#', 1)[0].strip().lower() for l in LOCAL_TERMS.read_text().splitlines())
    return {l for l in lines if l}


def internal_hits(text: str, terms):
    """(offset, word) for every word, host suffix or word pair on the local list."""
    words = [(m.start(), m.group(0).strip('.-')) for m in WORD.finditer(text.lower())]
    for i, (pos, w) in enumerate(words):
        cands = {w}
        parts = w.split('.')
        cands |= {'.'.join(parts[k:]) for k in range(1, len(parts))}   # a.b.corp.org -> corp.org
        if i + 1 < len(words):
            cands.add(f'{w} {words[i + 1][1]}')
        for c in cands & terms:
            yield pos, c


def _candidates():
    """What a push would publish: tracked files plus new ones not yet ignored.
    Gitignored local files (configs/local/, *.local.*) hold this machine's paths
    on purpose and never leave it. Outside a git checkout, scan everything."""
    try:
        out = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                             cwd=ROOT, capture_output=True, check=True).stdout
        return [ROOT / r for r in out.decode().split('\0') if r]
    except (OSError, subprocess.CalledProcessError):
        return list(ROOT.rglob('*'))


def files():
    for p in _candidates():
        if not p.is_file() or p.suffix.lower() in BINARY:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        rel = str(p.relative_to(ROOT))
        if rel in EXEMPT:
            continue
        yield p, rel


@pytest.mark.parametrize('kind', sorted(PATTERNS) + ['internal names'])
def test_no_leaks(kind):
    terms = local_terms() if kind not in PATTERNS else set()
    if kind not in PATTERNS and not terms:
        pytest.skip('no tests/leak_terms.local.txt on this machine')
    hits = []
    for p, rel in files():
        try:
            text = p.read_text(errors='replace')
        except OSError:
            continue
        found = (((m.start(), m.group(0)) for m in PATTERNS[kind].finditer(text))
                 if kind in PATTERNS else internal_hits(text, terms))
        for pos, s in found:
            line = text[:pos].count('\n') + 1
            hits.append(f'{rel}:{line}: {s[:60]}')
    assert not hits, f'{kind} found:\n  ' + '\n  '.join(hits[:20])
