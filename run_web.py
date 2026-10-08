"""Run the local React interface and Python API together during development."""

from __future__ import annotations

import argparse
import errno
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _check_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            # Match server sockets so recently closed test/dev connections do
            # not look like a live listener during a quick restart.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((HOST, port))
            probe.listen(1)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise RuntimeError(f"Cannot bind to {HOST}:{port}: {exc}") from exc
            raise RuntimeError(
                f"Port {port} is already in use. Stop the other service or select "
                "a different --api-port / --frontend-port."
            ) from exc


def _wait_until_ready(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"The service for {url} exited during startup.")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"The service for {url} did not become ready within 30 seconds.")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        # The API gives managed workers time to cancel and stop their FFmpeg
        # children before it exits. Do not kill that coordinator prematurely.
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=_port, default=8000)
    parser.add_argument("--frontend-port", type=_port, default=5173)
    parser.add_argument("--node", help="Node.js executable path (default: node on PATH)")
    parser.add_argument("--reload", action="store_true", help="Reload the API after Python edits")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the interface automatically")
    args = parser.parse_args(argv)

    def request_shutdown(_signal: int, _frame: object) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)

    processes: list[subprocess.Popen[bytes]] = []
    try:
        if args.api_port == args.frontend_port:
            raise RuntimeError("The API and frontend require different ports.")
        for module in ("uvicorn", "fastapi"):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(
                    f"{module} is missing from this Python environment. Activate "
                    ".venv and run python -m pip install -r requirements.txt."
                )
        node = shutil.which(args.node or "node")
        if node is None:
            raise RuntimeError("Node.js was not found. Install Node.js or pass --node /path/to/node.")
        vite = ROOT / "frontend" / "node_modules" / "vite" / "bin" / "vite.js"
        if not vite.is_file():
            raise RuntimeError("Frontend dependencies are missing. Run npm install in frontend first.")
        _check_port(args.api_port)
        _check_port(args.frontend_port)

        api_url = f"http://{HOST}:{args.api_port}"
        frontend_url = f"http://{HOST}:{args.frontend_port}"
        child_env = os.environ.copy()
        child_env["ROUNDNET_API_PORT"] = str(args.api_port)
        child_env["ROUNDNET_FRONTEND_PORT"] = str(args.frontend_port)
        child_env["VITE_API_URL"] = api_url
        child_env["PATH"] = str(Path(node).resolve().parent) + os.pathsep + child_env.get("PATH", "")
        api_command = [
            sys.executable, "-m", "uvicorn", "backend.app:app",
            "--host", HOST, "--port", str(args.api_port),
        ]
        if args.reload:
            api_command.append("--reload")
        print(f"Starting Roundnet API at {api_url}", flush=True)
        api = subprocess.Popen(api_command, cwd=ROOT, env=child_env, start_new_session=True)
        processes.append(api)
        _wait_until_ready(api_url + "/api/health", api)

        frontend = subprocess.Popen(
            [node, str(vite), "--host", HOST, "--port", str(args.frontend_port), "--strictPort"],
            cwd=ROOT / "frontend", env=child_env, start_new_session=True,
        )
        processes.append(frontend)
        _wait_until_ready(frontend_url, frontend)
        print(f"Roundnet is ready at {frontend_url}. Press Ctrl+C to stop both services.", flush=True)
        if not args.no_browser:
            webbrowser.open(frontend_url)
        while True:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"A service exited with status {process.returncode}.")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nStopping Roundnet services.", flush=True)
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"Roundnet could not start: {exc}", file=sys.stderr)
        return 1
    finally:
        for process in reversed(processes):
            _stop(process)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
