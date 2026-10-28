"""Read only the small exported records, never execute checkpoint code."""
import hashlib
import json
from pathlib import Path
import zipfile
from suffix_lab.store import digest

NAMES = ("long64-manifest.json", "preflight.json", "runtime.json", "prompts.jsonl",
         "rounds.jsonl", "references.jsonl")
PINNED_REVISION = "39890af682c068a51f5788594195f86e79a28575"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.hashes = {}
        self.raw = {}
        archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        try:
            for name in NAMES:
                if archive:
                    matches = [n for n in archive.namelist() if Path(n).name == name]
                    if len(matches) != 1:
                        raise ValueError(f"Missing/ambiguous {name} in {self.path}")
                    if archive.getinfo(matches[0]).file_size > 300*2**20:
                        raise ValueError("Unexpected exported record size")
                    data = archive.read(matches[0])
                else:
                    data = (self.path/name).read_bytes()
                self.hashes[name] = hashlib.sha256(data).hexdigest()
                self.raw[name] = data
        finally:
            if archive:
                archive.close()
        self.identity = digest(self.hashes)
        self.manifest = self.json("long64-manifest.json")
        self.preflight = self.json("preflight.json")
        self.runtime = self.json("runtime.json")
        self.prompts = self.lines("prompts.jsonl")
        self.rounds = self.lines("rounds.jsonl")

    def json(self, name):
        return json.loads(self.raw[name])

    def lines(self, name):
        return [json.loads(x) for x in self.raw[name].splitlines() if x.strip()]


def discover(root):
    root = Path(root)
    if root.is_file() or (root/"long64-manifest.json").exists():
        return Checkpoint(root)
    candidates = sorted({p.parent for p in root.rglob("long64-manifest.json")}, key=str)
    candidates += sorted(root.rglob("*.zip"))
    found = {}
    for path in candidates:
        try:
            cp = Checkpoint(path)
            if cp.manifest.get("code_revision") == PINNED_REVISION and len(cp.rounds) == 4050:
                found[cp.identity] = cp
        except (ValueError, KeyError, OSError, zipfile.BadZipFile):
            continue
    if len(found) != 1:
        raise ValueError(f"Need one completed guarded B64 checkpoint under {root}; found {len(found)}. "
                         "Attach long64_checkpoint.zip as Kaggle input, or set INPUT_ROOT to its unpacked folder.")
    return next(iter(found.values()))
