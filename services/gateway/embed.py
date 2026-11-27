"""Refresh the offline copy of the model and data inside static/index.html.

    python services/gateway/embed.py

Run after training so the page still scores correctly when opened from disk
with no services running. Reads browser_model.json and dashboard.json from the
CURRENT artifact version.
"""
import json
import re
from pathlib import Path

from loan_features import current_dir

INDEX = Path(__file__).resolve().parent / "static" / "index.html"


def main():
    version = current_dir()
    html = INDEX.read_text(encoding="utf-8")
    for key, file in (("MODEL", "browser_model.json"), ("DATA", "dashboard.json")):
        pattern = rf"/\*{key}\*/.*?/\*END\*/"
        if not re.search(pattern, html, flags=re.S):
            raise SystemExit(f"index.html has no /*{key}*/ ... /*END*/ marker")
        value = json.loads((version / file).read_text(encoding="utf-8"))
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        html = re.sub(pattern, lambda _: f"/*{key}*/{payload}/*END*/", html, count=1, flags=re.S)
    INDEX.write_text(html, encoding="utf-8")
    print(f"Embedded {version.name} into {INDEX}")


if __name__ == "__main__":
    main()
