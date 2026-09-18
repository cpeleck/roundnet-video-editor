"""Application entry point for Roundnet Rally Editor."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _configure_logging() -> None:
    """Configure a useful console log without creating files beside the video."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _missing_dependency_message(exc: ModuleNotFoundError) -> str:
    package = exc.name or "a required package"
    return (
        f"Roundnet Rally Editor could not start because {package!r} is not "
        "installed.\n\nCreate/activate the project virtual environment and run:\n"
        "    python -m pip install -r requirements.txt\n\n"
        "FFmpeg must also be installed separately (on macOS: brew install ffmpeg)."
    )


def main() -> int:
    """Create the Qt application and show its main window."""

    _configure_logging()

    try:
        # Apply the macOS Qt-plugin visibility guard before importing Qt;
        # Qt can cache an empty plugin-directory scan during its first import.
        from ui.main_window import MainWindow

        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ModuleNotFoundError as exc:
        message = _missing_dependency_message(exc)
        print(message, file=sys.stderr)
        return 2

    QCoreApplication.setOrganizationName("Roundnet Rally Editor")
    QCoreApplication.setApplicationName("Roundnet Rally Editor")
    QCoreApplication.setApplicationVersion("0.1.0")

    app = QApplication(sys.argv)
    app.setApplicationDisplayName("Roundnet Rally Editor")

    def handle_exception(
        exception_type: type[BaseException],
        exception: BaseException,
        exception_traceback: object,
    ) -> None:
        if exception_type is KeyboardInterrupt:
            sys.__excepthook__(exception_type, exception, exception_traceback)  # type: ignore[arg-type]
            return

        details = "".join(
            traceback.format_exception(
                exception_type,
                exception,
                exception_traceback,  # type: ignore[arg-type]
            )
        )
        logging.getLogger(__name__).critical("Unhandled exception\n%s", details)
        QMessageBox.critical(
            None,
            "Unexpected error",
            "Roundnet Rally Editor encountered an unexpected error.\n\n"
            f"{exception}\n\nDetails were written to the terminal.",
        )

    sys.excepthook = handle_exception

    window = MainWindow()
    window.show()
    logging.getLogger(__name__).info(
        "Started Roundnet Rally Editor from %s", Path.cwd()
    )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
