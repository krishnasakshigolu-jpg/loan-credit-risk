"""Where the trainer publishes model versions and where services find them.

    <ARTIFACTS_DIR>/
        v20261127T101500Z/   pipeline.joblib, metrics.json, dashboard.json, browser_model.json
        CURRENT              name of the version services should load

The trainer writes a full version folder first and only then swaps CURRENT,
so a service never sees a half-written model.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def artifacts_root() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", REPO_ROOT / "artifacts")).resolve()


def current_dir(root: Path | None = None) -> Path:
    root = root or artifacts_root()
    pointer = root / "CURRENT"
    if not pointer.exists():
        raise SystemExit(f"No model published in {root}. Run `python services/trainer/train.py` first.")
    version = root / pointer.read_text(encoding="utf-8").strip()
    if not version.is_dir():
        raise SystemExit(f"{pointer} points at {version.name}, which does not exist. Retrain.")
    return version


def publish(root: Path, version: str) -> None:
    tmp = root / "CURRENT.tmp"
    tmp.write_text(version, encoding="utf-8")
    os.replace(tmp, root / "CURRENT")
