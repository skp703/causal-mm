"""
Launch the FCM Project Visualizer GUI.

Usage (after pip install):
    causal-mm-gui              # launch on default port 8501
    causal-mm-gui --port 8502  # launch on custom port

Usage (from source):
    python -m causal_mm.gui.launch
    streamlit run gui/app.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _get_app_path() -> Path:
    """Return the absolute path to app.py inside the installed package."""
    return Path(__file__).resolve().parent / "app.py"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="causal-mm-gui",
        description="Launch the FCM Project Visualizer (Streamlit GUI)",
    )
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Port to serve the app on (default: 8501)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't automatically open a browser tab",
    )
    args = parser.parse_args(argv)

    app_path = _get_app_path()
    if not app_path.exists():
        print(f"Error: GUI app not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
    ]
    if not args.no_browser:
        cmd.extend(["--server.headless", "false"])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(
            "Error: streamlit not found. Install GUI dependencies:\n"
            "  pip install causal-mm[gui]",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
