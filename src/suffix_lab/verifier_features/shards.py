"""Atomic, immutable compressed tensor shards; no untrusted pickle loading."""
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile
from suffix_lab.store import atomic_json, canonical
from .checkpoint_loader import sha_file


def safe_name(name):
    path = PurePosixPath(name)
    return not (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name)


def restore(root, output):
    root, output = Path(root), Path(output)
    if root.is_file():
        with zipfile.ZipFile(root) as z:
            names = z.namelist()
            if "feature-manifest.json" not in names or any(not safe_name(n) for n in names):
                raise ValueError("Not a safe verifier feature archive")
            if sum(x.file_size for x in z.infolist()) > 24*2**30:
                raise ValueError("Unexpected feature archive size")
            # Copy only records needed to resume; labels/reports are rebuilt.
            for name in names:
                if name == "feature-manifest.json" or (name.startswith("shards/part-") and name.endswith(".zip")):
                    dst = output/name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    data = z.read(name)
                    if dst.exists():
                        import hashlib
                        if sha_file(dst) != hashlib.sha256(data).hexdigest():
                            raise ValueError("Conflicting local/resume shard")
                    else:
                        tmp = dst.with_suffix(dst.suffix+".tmp")
                        tmp.write_bytes(data)
                        os.replace(tmp, dst)
        return
    if not (root/"feature-manifest.json").exists():
        matches = list(root.rglob("feature-manifest.json"))
        if len(matches) != 1:
            raise ValueError("Set RESUME_ROOT to one unpacked feature output or feature checkpoint ZIP")
        root = matches[0].parent
    for src in [root/"feature-manifest.json", *sorted((root/"shards").glob("part-*.zip"))]:
        dst = output/src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha_file(src) != sha_file(dst):
                raise ValueError("Conflicting resume file")
        else:
            shutil.copyfile(src, dst)


class Shards:
    def __init__(self, output, identity):
        self.output, self.identity = Path(output), identity
        self.directory = self.output/"shards"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records, self.parts, self.pending = {}, [], []
        for part in sorted(self.directory.glob("part-*.zip")):
            with zipfile.ZipFile(part) as z:
                if json.loads(z.read("identity.json")) != identity:
                    raise ValueError("Shard identity differs")
                for raw in z.read("records.jsonl").splitlines():
                    record = json.loads(raw)
                    key = record["event"]["id"]
                    if key in self.records:
                        raise ValueError("Duplicate completed event")
                    payload = z.read(record["tensor_file"])
                    import hashlib
                    if hashlib.sha256(payload).hexdigest() != record["tensor_sha256"]:
                        raise ValueError("Corrupted tensor shard")
                    self.records[key] = record
            self.parts.append({"path": part.relative_to(self.output).as_posix(),
                               "sha256": sha_file(part), "bytes": part.stat().st_size})
        self.index()

    def index(self):
        atomic_json(self.output/"shards-index.json", {"identity": self.identity, "parts": self.parts,
                    "completed_events": len(self.records), "durability": "atomic per shard; final/interrupt flush"})

    def add(self, record, tensors):
        if record["event"]["id"] in self.records or any(r[0]["event"]["id"] == record["event"]["id"] for r in self.pending):
            raise ValueError("Duplicate event submission")
        self.pending.append((record, tensors))

    def flush(self):
        if not self.pending:
            return
        from safetensors.torch import save
        import hashlib
        started = time.perf_counter()
        part = self.directory/f"part-{len(self.parts):05d}.zip"
        if part.exists():
            raise ValueError("Refusing to overwrite immutable shard")
        temporary = part.with_suffix(".zip.tmp")
        records = []
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            z.writestr("identity.json", canonical(self.identity))
            for record, tensors in self.pending:
                tensor_file = "tensors/"+hashlib.sha256(record["event"]["id"].encode()).hexdigest()[:24]+".safetensors"
                data = save({k: v.contiguous() for k, v in tensors.items()})
                record = {**record, "tensor_file": tensor_file,
                          "tensor_shard": part.relative_to(self.output).as_posix(),
                          "tensor_sha256": hashlib.sha256(data).hexdigest()}
                z.writestr(tensor_file, data)
                records.append(record)
            z.writestr("records.jsonl", "".join(canonical(r)+"\n" for r in records))
        os.replace(temporary, part)
        self.parts.append({"path": part.relative_to(self.output).as_posix(), "sha256": sha_file(part),
                           "bytes": part.stat().st_size, "packing_wall_seconds": time.perf_counter()-started})
        self.records.update({r["event"]["id"]: r for r in records})
        self.pending.clear()
        self.index()
        print(f"[saved] {len(self.records)} events, {part.name}, {part.stat().st_size/2**20:.1f} MiB", flush=True)

    def archive(self):
        self.flush()
        archive = self.output/"verifier_features_checkpoint.zip"
        paths = [p for p in self.output.rglob("*") if p.is_file()
                 and p != archive and not p.name.endswith(".tmp")]
        if shutil.disk_usage(self.output).free < sum(p.stat().st_size for p in paths)+512*2**20:
            raise OSError("Not enough disk for final combined ZIP; all immutable shards and metadata remain available")
        temporary = archive.with_suffix(".zip.tmp")
        # Shards are already compressed. Store them without recompressing GBs.
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in paths:
                z.write(p, p.relative_to(self.output).as_posix())
        os.replace(temporary, archive)
        return archive


def load_event_tensors(output, event):
    """Read an events.jsonl tensor descriptor safely for later analysis."""
    from safetensors.torch import load
    import hashlib
    descriptor = event["tensors"]
    if not safe_name(descriptor["shard"]) or not safe_name(descriptor["member"]):
        raise ValueError("Unsafe tensor descriptor")
    with zipfile.ZipFile(Path(output)/descriptor["shard"]) as z:
        data = z.read(descriptor["member"])
    if hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
        raise ValueError("Tensor descriptor checksum differs")
    return load(data)
