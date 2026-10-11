"""
src/console.py
--------------
One job: make the demo and experiment scripts printable on a default
Windows console.

THE PROBLEM
===========
Every report in this project draws box-drawing characters (``═ ─ │``),
arrows and the em-dash used for "not fitted". On Windows, Python picks
the console's ANSI code page (cp1252) for ``sys.stdout`` when output is
not already UTF-8, and printing any of those characters raises::

    UnicodeEncodeError: 'charmap' codec can't encode characters ...

so a demo dies partway through its own report. The workaround was to set
``PYTHONIOENCODING=utf-8`` before every run, which is easy to forget and
silently turns a working script into a crashing one.

THE FIX
=======
``enable_unicode_stdout()`` reconfigures ``sys.stdout`` / ``sys.stderr``
to UTF-8 in place (``io.TextIOWrapper.reconfigure``, Python 3.7+). Call
it once at the top of ``main()``. It is:

- idempotent — calling it twice is harmless;
- a no-op where the stream is already UTF-8 (Linux, macOS, or Windows
  with ``PYTHONIOENCODING`` already set);
- non-fatal — if a stream cannot be reconfigured (replaced by a non-text
  object, as pytest's ``capsys`` does), it returns ``False`` and the
  caller carries on rather than crashing on a cosmetic concern.

``errors="replace"`` is deliberate: a character that still cannot be
encoded prints as ``?`` instead of killing a long experiment run.

Worked example
==============
    >>> from src.console import enable_unicode_stdout
    >>> def main() -> None:
    ...     enable_unicode_stdout()
    ...     print("═" * 10)        # safe on a cp1252 console
"""

from __future__ import annotations

import logging
import sys
from typing import IO, Any

logger = logging.getLogger(__name__)

_TARGET_ENCODING = "utf-8"


def _reconfigure(stream: IO[Any] | None) -> bool:
    """Reconfigure one stream to UTF-8. True if it is UTF-8 afterwards."""
    if stream is None:
        return False
    current = (getattr(stream, "encoding", "") or "").lower().replace("_", "-")
    if current in ("utf-8", "utf8"):
        return True
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        # Not a TextIOWrapper — e.g. pytest's capture object. Nothing to do.
        return False
    try:
        reconfigure(encoding=_TARGET_ENCODING, errors="replace")
    except (ValueError, OSError, AttributeError) as exc:
        logger.debug("Could not reconfigure %r to UTF-8: %s", stream, exc)
        return False
    return True


def enable_unicode_stdout() -> bool:
    """
    Make ``sys.stdout`` and ``sys.stderr`` accept the report characters.

    Returns
    -------
    bool
        True if stdout is UTF-8 once this returns (already was, or was
        successfully reconfigured). Never raises.
    """
    ok = _reconfigure(sys.stdout)
    _reconfigure(sys.stderr)
    return ok
