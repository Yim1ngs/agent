from pathlib import Path
import re

def test_candidate_flag():
    p = Path(__file__).resolve().parents[1] / "workspace" / "candidate_flag.txt"
    assert p.exists(), "candidate_flag.txt not found"
    flag = p.read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"flag\{[A-Za-z0-9_\-]+\}", flag), f"invalid flag format: {flag!r}"