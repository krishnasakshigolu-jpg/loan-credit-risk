"""Run the whole stack on this machine.

    python scripts/dev.py            train if no model is published, then start every service
    python scripts/dev.py --train    retrain first, even if a model exists
    python scripts/dev.py --open     also open the dashboard in the browser

Starts, each as its own process:
    scoring   http://127.0.0.1:8001
    insights  http://127.0.0.1:8002
    gateway   http://127.0.0.1:8000   <- open this one

Ctrl+C stops all of them. If any service exits, the others are stopped too.

Only the gateway is meant to be reached from outside. Set PORT and HOST to move
it (Render sets PORT; render.yaml sets HOST=0.0.0.0). The other services always
listen on 127.0.0.1.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
GATEWAY_PORT = int(os.environ.get("PORT", 8000))
GATEWAY_HOST = os.environ.get("HOST", "127.0.0.1")
SERVICES = [  # name, folder, module, host, port
    ("scoring", "services/scoring", "scoring_service.main:app", "127.0.0.1", 8001),
    ("insights", "services/insights", "insights_service.main:app", "127.0.0.1", 8002),
    ("gateway", "services/gateway", "gateway_service.main:app", GATEWAY_HOST, GATEWAY_PORT),
]
URL = f"http://127.0.0.1:{GATEWAY_PORT}"


def wait_healthy(url, procs, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(p.poll() is not None for p in procs.values()):
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="retrain before starting")
    ap.add_argument("--open", action="store_true", help="open the dashboard in the browser")
    args = ap.parse_args()
    # Treat SIGTERM (Render, CI, `kill`) like Ctrl+C so the child services are stopped too.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    env = {**os.environ, "ARTIFACTS_DIR": str(ARTIFACTS),
           "SCORING_URL": "http://127.0.0.1:8001", "INSIGHTS_URL": "http://127.0.0.1:8002"}

    if args.train or not (ARTIFACTS / "CURRENT").exists():
        print("Training the model...")
        subprocess.run([sys.executable, "services/trainer/train.py"], cwd=ROOT, env=env, check=True)

    procs = {}
    try:
        for name, folder, module, host, port in SERVICES:
            procs[name] = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", module, "--host", host, "--port", str(port)],
                cwd=ROOT / folder, env=env)
        if not wait_healthy(f"{URL}/healthz", procs):
            print("The services did not come up. See the errors above.")
            return 1
        print(f"\nDashboard: {URL}   (Ctrl+C to stop)\n")
        if args.open:
            webbrowser.open(URL)
        while all(p.poll() is None for p in procs.values()):
            time.sleep(0.5)
        dead = [n for n, p in procs.items() if p.poll() is not None]
        print(f"{', '.join(dead)} stopped, shutting down the rest.")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
