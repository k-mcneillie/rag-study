"""The chat interface loads under Chainlit's module loader.

Chainlit does not import ``app/main.py`` — it reads the file, builds a module
from a spec, and executes it *without* registering it in :data:`sys.modules`.
Anything that resolves its own annotations by looking its defining module up
there fails at import time, with a traceback that names ``dataclasses`` rather
than the line responsible. A ``@dataclass`` in that file did exactly that.

Ordinary ``import`` succeeds where Chainlit fails, so importing the module the
usual way would not have caught it. This reproduces Chainlit's loader instead.

The test is skipped when Chainlit is not installed, which is the case in CI:
the interface is an optional extra.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("chainlit", reason="The chat interface is an optional extra.")

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_the_app_executes_outside_sys_modules() -> None:
    """The interface imports the way Chainlit actually loads it.

    Mirrors ``chainlit.config.load_module``: build a module from a file
    location and execute it, leaving ``sys.modules`` untouched.
    """
    spec = importlib.util.spec_from_file_location("rag_chat_app", APP)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    assert module.__name__ not in sys.modules

    spec.loader.exec_module(module)

    assert callable(module.build_retrieval_orchestrator)
    assert callable(module.build_chat_model)
