"""The Chainlit entry module loads under Chainlit's own loader.

Chainlit does not ``import`` ``main.py`` — it reads the file, builds a module
from a spec, and executes it *without* registering it in :data:`sys.modules`.
Anything that resolves its own annotations by looking its defining module up
there fails at import time, with a traceback that names ``dataclasses`` rather
than the line responsible. That is why ``_Turn`` and ``_StreamState`` are
hand-written classes, not dataclasses.

Ordinary ``import`` succeeds where Chainlit fails, so importing the module the
usual way would not catch a regression. This reproduces Chainlit's loader.

The test is skipped when Chainlit is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("chainlit", reason="The chat interface needs Chainlit.")

MAIN = Path(__file__).resolve().parents[1] / "src" / "rag_chat" / "main.py"


def test_the_entry_module_executes_outside_sys_modules() -> None:
    """The interface loads the way Chainlit actually loads it.

    Mirrors ``chainlit.config.load_module``: build a module from a file
    location and execute it, leaving ``sys.modules`` untouched.
    """
    spec = importlib.util.spec_from_file_location("rag_chat_entry", MAIN)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    assert module.__name__ not in sys.modules

    spec.loader.exec_module(module)

    assert callable(module.start)
    assert callable(module.answer)
    assert isinstance(module._Turn, type)
