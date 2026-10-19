"""Offline paired-prefix diagnostics and independent fixed-horizon repairs."""
from collections import defaultdict
from dataclasses import asdict
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import time
import zipfile
from .assets import ROOT, TARGET, AR, DLLM, source_digest
from .metrics import suffix_metrics, redraft_metrics, prefix_length
from .selection import Observation, select_keep, oracle_masks
from .store import Store, atomic_json, digest


def offsets(prompt_length, reference_length, horizon=32, count=5):
    """Shared beginning/middle states, including a valid absolute aligned state."""
    last = min(reference_length-horizon-1, reference_length*2//3)
    if last < 0:
        return []
    choices = [0]
    aligned = (-prompt_length) % 32
    if aligned <= last:
        choices.append(aligned)
    if count > 1:
        choices.extend(round(last*i/(count-1)) for i in range(1, count))
    choices.extend(range(last+1))
    unique = list(dict.fromkeys(choices))[:count]
    return sorted(unique)


def stratified_events(records, count, seed=42, sources=("dllm_top1", "ar_greedy")):
    """Frozen sample before evaluating any repair; dLLM gets 2:1 priority."""
    strata = defaultdict(list)
    for row in records:
        r = row["metrics"]["first_rejection"]
        if 1 <= r < len(row["draft"]["tokens"]) and row["source"] in sources:
            strata[(row["source"], row["dataset"], (r-1)//8)].append(row["id"])
    rng = random.Random(seed)
    for key in sorted(strata):
        strata[key].sort()
        rng.shuffle(strata[key])
    picked = []
    usage = defaultdict(int)
    cycle = ["dllm_top1", "dllm_top1", "ar_greedy"]
    while len(picked) < count and any(strata.values()):
        for source in cycle:
            keys = [k for k in sorted(strata) if k[0] == source and strata[k]]
            if not keys:
                continue
            # Round robin across benchmark/rejection bins, preserving source priority.
            key = min(keys, key=lambda k: (usage[k], k))
            picked.append(strata[key].pop())
            usage[key] += 1
            if len(picked) >= count:
                break
    return picked


def nonoracle_plans(obs, config, seed):
    """No reference is accepted by this function, including for natural agreement."""
    plans = [("fresh_L", []), ("direct_rejoin", list(range(len(obs.old_tokens))))]
    for ratio in config["keep_ratios"]:
        for method in ("drafter_confidence", "verifier_confidence", "old_agreement_ranked"):
            plans.append((f"{method}@{ratio:g}", select_keep(obs, method, ratio, seed)))
        for repeat in range(config["random_repeats"]):
            plans.append((f"random@{ratio:g}#{repeat}", select_keep(obs, "random", ratio, seed+repeat)))
    natural = select_keep(obs, "old_agreement", 0, seed)
    plans.append(("old_agreement", natural))
    ratio = len(natural)/len(obs.old_tokens)
    for repeat in range(config["random_repeats"]):
        plans.append((f"random_matched_agreement#{repeat}", select_keep(obs, "random", ratio, seed+repeat)))
    return plans


def pack(output):
    """Snapshot live SQLite with its backup API; never zip WAL inconsistently."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    db = output / "run.sqlite"
    if db.exists():
        store = Store(db)
        store.backup(output / "checkpoint.sqlite")
        for kind in ("reference", "test1", "test2"):
            store.export(kind, output / (kind+".jsonl"))
        store.close()
    path = output / "suffix_checkpoint.zip"
    temp = output / "suffix_checkpoint.tmp"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in output.rglob("*"):
            if file.is_file() and file.name not in ("run.sqlite", "run.sqlite-wal", "run.sqlite-shm", path.name, temp.name):
                archive.write(file, file.relative_to(output).as_posix())
    if temp.stat().st_size >= 2_000_000_000:
        raise RuntimeError("Artifact exceeds 2GB; inspect outputs before continuing")
    temp.replace(path)
    return path


def restore(archive, output):
    """Only restore into an empty output, rejecting traversal/symlink entries."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Restore requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            dest = (output / entry.filename).resolve()
            if not dest.is_relative_to(output) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe checkpoint archive member")
        z.extractall(output)
    if (output / "checkpoint.sqlite").exists():
        import shutil
        shutil.copy2(output / "checkpoint.sqlite", output / "run.sqlite")


def environment():
    import torch
    packages = {dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                         text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        driver = ["unavailable"]
    return {"packages": packages, "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "commit": commit, "driver": driver, "attention": "sdpa", "batch_size": 1}


class Runner:
    def __init__(self, config, manifest, output, stage="all"):
        from .runtime import tokenizers, load_target, ARSession
        self.config, self.manifest = config, manifest
        self.output, self.stage = Path(output), stage
        self.output.mkdir(parents=True, exist_ok=True)
        preflight = json.loads((self.output / "preflight.json").read_text())
        if (not preflight.get("passed") or preflight["config"] != config or
                preflight["assets_digest"] != manifest["content_digest"] or preflight.get("source_digest") != source_digest()):
            raise ValueError("Matching real-model preflight required; run preflight first")
        self.tokenizer, token_evidence = tokenizers(manifest["paths"])
        import torch
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        self.target, precision, _ = load_target(manifest["paths"][TARGET], config, preflight["precision"])
        self.verifier = ARSession(self.target)
        self.verifier.draft(self.tokenizer.encode("Warmup: 1 + 1 ="), 2)
        self.env = environment()
        identity = {"protocol": "suffix-top1-v1", "config": config,
                    "source_digest": source_digest(),
                    "asset_digest": manifest["content_digest"], "target_precision": precision,
                    "tokenizer": token_evidence, "environment": self.env}
        self.store = Store(self.output / "run.sqlite", identity)
        atomic_json(self.output / "run-manifest.json", identity)
        self.deadline = time.monotonic() + config["wall_hours"] * 3600
        self.completed_units = 0

    def checkpoint(self):
        self.completed_units += 1
        if self.completed_units % 25 == 0:
            pack(self.output)
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Soft session wall budget reached; resume the same output directory")

    def references(self):
        c = self.config
        eos = self.target.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for prompt in self.manifest["prompts"]:
            if self.store.get("reference", prompt["id"]):
                continue
            ids = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt["question"]}],
                    tokenize=True, add_generation_prompt=True)
            if len(ids) > c["max_prompt_tokens"]:
                raise ValueError(f"Prompt exceeds configured context limit: {prompt['id']}; do not truncate silently")
            draft = self.verifier.draft(ids, c["reference_tokens"], eos)
            row = {**prompt, "prompt_ids": ids, "tokens": draft.tokens, "stats": draft.stats,
                   "ended_eos": bool(draft.tokens and draft.tokens[-1] in eos),
                   "offsets": offsets(len(ids), len(draft.tokens), c["proposal_length"], c["prefixes_per_prompt"])}
            self.store.put("reference", prompt["id"], row)
            print("reference", prompt["id"], len(draft.tokens), "tokens", len(row["offsets"]), "states", flush=True)
            self.checkpoint()
        self.store.put("meta", "references_complete", True)

    def test1(self):
        from .runtime import ARSession, DLLMDrafter, load_model, release, memory
        c = self.config
        if not self.store.get("meta", "references_complete"):
            raise ValueError("Finish references before Test 1")
        for source, name in (("ar_greedy", AR), ("dllm_top1", DLLM)):
            if self.store.get("meta", source+"_complete"):
                continue
            model = load_model(self.manifest["paths"][name], "cuda:1")
            drafter = ARSession(model) if source == "ar_greedy" else DLLMDrafter(model, c)
            refs = self.store.rows("reference")
            if refs:
                warmup = drafter.draft(refs[0]["prompt_ids"], 2)
                self.store.put("warmup", source, warmup.stats)
            self.store.put("hardware", source, memory())
            try:
                for ref in self.store.rows("reference"):
                    for offset in ref["offsets"]:
                        state_id = f"{ref['id']}:{offset}"
                        key = f"{state_id}:{source}"
                        if self.store.get("test1", key):
                            continue
                        prefix = ref["prompt_ids"] + ref["tokens"][:offset]
                        gold = ref["tokens"][offset:offset+c["proposal_length"]]
                        draft = drafter.draft(prefix, c["proposal_length"])
                        feedback = self.verifier.score(prefix, draft.tokens, c["top_k"])
                        m = suffix_metrics(draft.tokens, gold, feedback["top1"])
                        old_k = prefix_length(draft.tokens, feedback["top1"])
                        # Before first mismatch draft/true prefixes coincide. Parallel vs
                        # incremental numerical disagreement invalidates this experiment.
                        if old_k != m["accepted_prefix"] or (old_k < len(gold) and feedback["top1"][old_k] != gold[old_k]):
                            self.store.put("failure", key, {"reason": "causal/reference consistency", "feedback": feedback,
                                                          "draft": draft.tokens, "reference": gold})
                            raise AssertionError(f"Target verification/reference disagree at {key}; inspect precision/kernels")
                        row = {"id": key, "state_id": state_id, "prompt_id": ref["id"],
                               "dataset": ref["dataset"], "source": source, "offset": offset,
                               "context_length": len(prefix), "physical_phase": len(prefix) % 32,
                               "draft": asdict(draft), "feedback": feedback, "reference": gold, "metrics": m}
                        eos_id = self.target.generation_config.eos_token_id if hasattr(self, "target") else []
                        eos_ids = eos_id if isinstance(eos_id, list) else [eos_id]
                        row["draft_first_eos"] = next((i for i, t in enumerate(draft.tokens) if t in eos_ids), None)
                        self.store.put("test1", key, row)
                        print("test1", key, "r=", m["first_rejection"], flush=True)
                        self.checkpoint()
                self.store.put("meta", source+"_complete", True)
            finally:
                del drafter, model
                release(None)
        self.store.put("meta", "test1_complete", True)

    def test2(self):
        from .runtime import DLLMDrafter, load_model, release
        c = self.config
        if not self.store.get("meta", "test1_complete"):
            raise ValueError("Finish Test 1 before freezing the Test 2 sample")
        selected = self.store.get("meta", "test2_sample")
        if selected is None:
            selected = stratified_events(self.store.rows("test1"), c["test2_events"], c["seed"], c["test2_sources"])
            self.store.put("meta", "test2_sample", selected)
        model = load_model(self.manifest["paths"][DLLM], "cuda:1")
        drafter = DLLMDrafter(model, c)
        refs = self.store.rows("reference")
        if refs:
            warmup = drafter.draft(refs[0]["prompt_ids"], 2)
            self.store.put("warmup", "test2", warmup.stats)
        try:
            for event_id in selected:
                if self.store.get("event_complete", event_id):
                    continue
                row = self.store.get("test1", event_id)
                ref = self.store.get("reference", row["prompt_id"])
                r = row["metrics"]["first_rejection"]  # 1-based, correction at r-1
                prefix = ref["prompt_ids"] + ref["tokens"][:row["offset"]+r]
                old, gold = row["draft"]["tokens"][r:], row["reference"][r:]
                obs = Observation(tuple(old), tuple(row["draft"]["confidence"][r:]),
                                  tuple(row["feedback"]["probability"][r:]), tuple(row["feedback"]["top1"][r:]))
                seed = int(digest([c["seed"], event_id])[:8], 16)
                start = time.perf_counter()
                plans = nonoracle_plans(obs, c, seed)
                selector_ms = (time.perf_counter()-start)*1000
                cached_masks = {}

                def arm(method, keep):
                    key = event_id+"|"+method
                    existing = self.store.get("test2", key)
                    mask_key = digest(keep)
                    if existing:
                        cached_masks[mask_key] = existing
                        return existing
                    # Independent cold replay: no suffix KV or preceding arm's mutable state.
                    previous = cached_masks.get(mask_key)
                    result = drafter.draft(prefix, len(old), {i: old[i] for i in keep}) if previous is None else None
                    metrics = redraft_metrics(result.tokens if result else previous["tokens"], gold, old, keep)
                    payload = {"id": key, "event_id": event_id, "prompt_id": row["prompt_id"],
                               "dataset": row["dataset"], "source": row["source"], "method": method,
                               "oracle": method.startswith("oracle"), "first_rejection": r, "horizon": len(old),
                               "context_length": len(prefix), "physical_phase": len(prefix) % 32,
                               "keep": keep, "keep_ratio": len(keep)/len(old),
                               "tokens": result.tokens if result else previous["tokens"],
                               "stats": result.stats if result else previous["stats"], "metrics": metrics,
                               "mask_reused_from": previous["id"] if previous else None,
                               "selector_ms_event": selector_ms, "latency_policy": "independent cold prefix replay"}
                    self.store.put("test2", key, payload)
                    cached_masks[mask_key] = payload
                    self.checkpoint()
                    return payload

                fresh = None
                for method, keep in plans:
                    value = arm(method, keep)
                    if method == "fresh_L":
                        fresh = value
                candidates = oracle_masks(old, gold, c["oracle_subset_limit"], seed)
                a = arm("oracle_A", candidates[0])
                tried = [a]
                if a["metrics"]["accepted_prefix"] <= fresh["metrics"]["accepted_prefix"]:
                    for i, keep in enumerate(candidates[1:]):
                        tried.append(arm(f"oracle_B#{i}", keep))
                best = max(tried, key=lambda x: (x["metrics"]["accepted_prefix"], x["metrics"]["full_block_accuracy"]))
                self.store.put("oracle_search", event_id, {"event_id": event_id, "candidates": len(tried),
                    "best_id": best["id"], "best_accepted_prefix": best["metrics"]["accepted_prefix"],
                    "oracle_A_id": a["id"], "acquisition_cost_excluded": True, "bounded_not_global": True})
                self.store.put("event_complete", event_id, True)
                print("test2", event_id, "fresh=", fresh["metrics"]["accepted_prefix"],
                      "oracle_A=", a["metrics"]["accepted_prefix"], flush=True)
            self.store.put("meta", "test2_complete", True)
        finally:
            del drafter, model
            release(None)

    def run(self):
        status = "complete"
        try:
            for name in ("references", "test1", "test2"):
                if self.stage in ("all", name):
                    getattr(self, name)()
        except TimeoutError as error:
            status = "partial"
            print(str(error), flush=True)
        except BaseException:
            status = "failed"
            raise
        finally:
            atomic_json(self.output / "status.json", {"status": status, "stage": self.stage,
                "references_complete": bool(self.store.get("meta", "references_complete")),
                "test1_complete": bool(self.store.get("meta", "test1_complete")),
                "test2_complete": bool(self.store.get("meta", "test2_complete"))})
            pack(self.output)
            self.store.close()
        return status
