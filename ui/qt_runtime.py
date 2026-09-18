"""Small compatibility guard for macOS development virtual environments."""

import logging
import os
from pathlib import Path
import stat
import sys


def prepare_qt_plugins() -> None:
    """Let Qt discover dylibs marked Finder-hidden by an environment manager.

    Qt's plugin directory scan excludes UF_HIDDEN files on macOS. Some managed
    virtual environments apply that flag to every dependency, making installed
    playback backends appear missing. Only clear this visibility flag on Qt's
    plugin libraries; do not change permissions, quarantine, or other flags.
    Read-only installations remain untouched and log a diagnostic.
    """
    if sys.platform != "darwin" or not hasattr(os, "chflags"):
        return
    import PySide6
    root = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    for folder in ("platforms", "multimedia", "imageformats"):
        for library in (root / folder).glob("*.dylib"):
            try:
                flags = library.stat().st_flags
                if flags & stat.UF_HIDDEN:
                    os.chflags(library, flags & ~stat.UF_HIDDEN)
            except OSError as exc:
                logging.getLogger(__name__).warning("Cannot expose Qt plugin %s: %s", library.name, exc)
