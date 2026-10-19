"""Prepare small text-only source tree for the authenticated GitHub connector."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FILES = [".gitignore", "README.md", "pyproject.toml", "requirements-kaggle.txt", "source-lock.json", "KAGGLE_CELLS.md", "KAGGLE_TWO_CELLS.md"]
files = [ROOT/name for name in FILES]
for directory in ("src", "tests", "scripts", "configs", "docs", "notebooks"):
    files.extend(p for p in (ROOT/directory).rglob("*") if p.is_file() and p.suffix in (".py", ".md", ".json", ".ipynb"))
entries = []
for file in sorted(set(files)):
    if "__pycache__" in file.parts or file.stat().st_size > 500_000:
        raise ValueError(f"Unexpected artifact: {file.name}")
    content = file.read_text(encoding="utf-8")
    if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,})", content):
        raise ValueError("Credential-shaped text found; refusing publication")
    entries.append({"path": file.relative_to(ROOT).as_posix(), "mode": "100644", "type": "blob", "content": content})
print(json.dumps(entries, ensure_ascii=True))
