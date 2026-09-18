"""PySide6 user interface for the Roundnet video editor."""

from .qt_runtime import prepare_qt_plugins

prepare_qt_plugins()

from .main_window import MainWindow

__all__ = ["MainWindow"]
