"""Test bootstrap.

Sets a dummy ``OPENAI_API_KEY`` before any ``src.*`` import so the Settings
singleton can construct in CI where no real key exists. Every test mocks the
network layer, so this key is never used for a real call (CLAUDE.md Phase 9:
never hit the real API in CI). ``setdefault`` so a real exported key is left
intact for local runs.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
