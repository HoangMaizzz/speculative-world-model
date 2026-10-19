"""Transactional per-unit checkpoints; resume refuses incompatible runs."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class Store:
    def __init__(self, path, identity=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, key TEXT, payload TEXT, PRIMARY KEY(kind,key))")
        if identity is not None:
            previous = self.get("meta", "identity")
            if previous is not None and digest(previous) != digest(identity):
                self.db.close()
                raise ValueError("Resume identity differs: use the matching assets/config/precision or a new output directory")
            self.put("meta", "identity", identity)

    def get(self, kind, key):
        row = self.db.execute("SELECT payload FROM records WHERE kind=? AND key=?", (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (kind, key, canonical(value)))

    def rows(self, kind):
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM records WHERE kind=? ORDER BY key", (kind,))]

    def export(self, kind, path):
        path = Path(path)
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w", encoding="utf-8") as f:
            for row in self.rows(kind):
                f.write(canonical(row) + "\n")
        os.replace(temp, path)

    def backup(self, path):
        with closing(sqlite3.connect(str(path))) as destination:
            self.db.backup(destination)

    def close(self):
        self.db.close()
