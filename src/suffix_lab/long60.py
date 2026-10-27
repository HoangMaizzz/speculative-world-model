"""Online dLLM60 -> AR greedy SD, followed by independent offline diagnostics."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import sqlite3
import time
import traceback
from types import SimpleNamespace
import zipfile

from .assets import ROOT, TARGET, DLLM, source_digest
from .store import Store, atomic_json, digest
from .long60_plan import decide, diagnose, until_eos, validate_draft


class BudgetReached(Exception):
    pass


def validate_config(c):
    if (c["draft_length"] != 60 or c["diagnostic_horizon"] != 59
            or c["target_precision"] != "fp16" or c["physical_block_size"] != 32
            or c["subblock_size"] != 8 or c["minimum_run"] < 2
            or c["reference_lookahead"] < c["draft_length"]+c["diagnostic_horizon"]
            or max(c["shift_radii"]) != 59 or min(c["shift_radii"]) < 1
            or not 1 <= c["prompts_per_dataset"] <= 100 or c["max_new_tokens"] < 1
            or c["soft_budget_seconds"] <= 0 or c["zip_interval_seconds"] <= 0):
        raise ValueError("Invalid fixed B60/FP16/native geometry or diagnostic configuration")


def prepare_prompts(c):
    from datasets import load_dataset
    lock = json.loads((ROOT/"source-lock.json").read_text())
    groups = {}
    for ds in c["datasets"]:
        name = {"gsm8k": "openai/gsm8k", "math500": "HuggingFaceH4/MATH-500"}[ds]
        kw = {"name": "main"} if ds == "gsm8k" else {}
        data = load_dataset(name, split="test", revision=lock["datasets"][name], **kw)
        indices = random.Random(c["seed"]).sample(range(len(data)), c["prompts_per_dataset"])
        groups[ds] = [{"id": f"{ds}:{i}", "dataset": ds, "index": i,
                       "question": data[i]["question" if ds == "gsm8k" else "problem"],
                       "dataset_fingerprint": data._fingerprint} for i in indices]
    return [groups[ds][i] for i in range(c["prompts_per_dataset"]) for ds in c["datasets"]]


def clear_unused():
    gc.collect()
    import sys
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def numerical_identity(runtime):
    env = runtime["environment"]
    return {"python": env["python"], "cuda": env["cuda"], "gpus": env["gpus"],
            "packages": {k: env["packages"].get(k) for k in
                         ("torch", "transformers", "accelerate", "tokenizers", "safetensors")},
            "tokenizer": runtime["tokenizer"]["shared_vocab_sha256"],
            "target_device_map": runtime["target_device_map"], "precision": runtime["target_precision"],
            "eos": runtime["effective_eos_ids"]}


class ParallelVerifier:
    def __init__(self, model):
        self.model = model

    def score(self, prefix, proposal):
        import torch
        from .runtime import sync_model
        sync_model(self.model)
        with torch.inference_mode():
            out = self.model(torch.tensor([prefix+proposal], device=self.model.device),
                             use_cache=False, logits_to_keep=len(proposal)+1)
            logits = out.logits[0].float()
            if logits.shape[0] != len(proposal)+1:
                raise AssertionError("Shifted verification/bonus row count differs")
            top1 = logits.argmax(-1)
            top2 = logits.topk(2, dim=-1).values
            result = {"top1_with_bonus": top1.tolist(),
                      "logit_margin": (top2[:, 0]-top2[:, 1]).tolist(),
                      "forwards": 1, "input_tokens": len(prefix)+len(proposal),
                      "cache_policy": "cold full causal prefix per round; no stale draft KV"}
        sync_model(self.model)
        return result


def load_bundle(c, output):
    import torch
    from .extension import prepare_models, audited_draft
    from .runtime import (configure_numerics, tokenizers, load_target, load_model,
                          ARSession, DLLMDrafter, target_consistency, memory)
    from .experiment import environment
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Requires two CUDA GPUs, approximately 16GB each")
    torch.manual_seed(c["seed"])
    configure_numerics(c)
    paths, revisions = prepare_models()
    tokenizer, tokenizer_evidence = tokenizers(paths)
    target, _, _ = load_target(paths[TARGET], c)
    # This is a headroom/correctness experiment. No mixed compute/wall speedup metric.
    target._suffix_exclude_peer_transfers = False
    ar = ARSession(target)
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": "Compute 17 + 28. Explain briefly."}],
                                           tokenize=True, add_generation_prompt=True)
    audit = {"passed": False, "target": target_consistency(ar, prefix, 60), "dllm": []}
    atomic_json(output/"preflight.json", audit)
    if not audit["target"]["passed"]:
        raise AssertionError("B60 target cached/parallel consistency failed; preflight.json preserves evidence")
    model = load_model(paths[DLLM], "cuda:1", precision="fp16")
    drafter = DLLMDrafter(model, c)
    for phase in (0, 1, 7, 8, 31):
        p = prefix+[prefix[-1]]*((phase-len(prefix)) % 32)
        print(f"[long60 preflight] native B60 cached/full phase={phase}", flush=True)
        result = audited_draft(drafter, p, 60, {})
        validate_draft(result, 60, p)
        audit["dllm"].append({"phase": phase, "cached_full_equal": True, "unmask_forwards": 60})
        atomic_json(output/"preflight.json", audit)
    eos = set()
    for obj in (tokenizer, target.config, target.generation_config, model.config, model.generation_config):
        value = getattr(obj, "eos_token_id", None)
        eos.update(value if isinstance(value, (list, tuple)) else [] if value is None else [value])
    if not eos or c["mask_id"] in eos:
        raise ValueError("Invalid EOS/MASK identity")
    audit.update(passed=True, effective_eos_ids=sorted(eos))
    atomic_json(output/"preflight.json", audit)
    runtime = {"environment": environment(), "models": revisions,
               "target_device_map": {k: str(v) for k, v in target.hf_device_map.items()},
               "target_precision": "fp16", "drafter_device": "cuda:1", "memory": memory(),
               "tokenizer": tokenizer_evidence, "effective_eos_ids": sorted(eos),
               "latency_study": False, "operational_wall_includes_peer_transfers": True}
    previous_runtime = output/"runtime.json"
    if previous_runtime.exists():
        previous = json.loads(previous_runtime.read_text(encoding="utf-8"))
        if numerical_identity(previous) != numerical_identity(runtime):
            error = ValueError("Resume GPU/packages/tokenizer numerical identity differs; previous runtime.json preserved")
            error.evidence = {"previous": numerical_identity(previous), "current": numerical_identity(runtime)}
            raise error
    atomic_json(output/"runtime.json", runtime)
    return SimpleNamespace(tokenizer=tokenizer, drafter=drafter, verifier=ParallelVerifier(target), eos=eos,
                           canonical=lambda p, n: asdict(ar.draft(p, n, eos)))


def generate_prompt(prompt, ids, c, store, bundle, boundary, checkpoint):
    previous = sorted([r for r in store.rows("round") if r["prompt_id"] == prompt["id"]], key=lambda r: r["round_index"])
    emitted = []
    for index, row in enumerate(previous):
        if (row["round_index"] != index or row["offset"] != len(emitted)
                or row["prefix_digest"] != digest(ids+emitted)):
            raise ValueError("Checkpoint round chain differs")
        expected = decide(row["proposal"], row["verification"]["top1_with_bonus"], bundle.eos,
                          c["max_new_tokens"]-len(emitted))
        if expected != row["decision"]:
            raise ValueError("Checkpoint commit decision differs")
        emitted.extend(row["decision"]["committed_tokens"])
        if emitted[-1] in bundle.eos and index != len(previous)-1:
            raise ValueError("Checkpoint has rounds after EOS")
    while len(emitted) < c["max_new_tokens"] and not (emitted and emitted[-1] in bundle.eos):
        boundary()
        offset, index = len(emitted), len(previous)
        key = f"{prompt['id']}:{index:05d}"
        prefix = ids+emitted
        if len(prefix)+c["draft_length"] > c["context_limit"]:
            raise ValueError("Context limit; prompt/output was not silently truncated")
        store.put("meta", "active", {"phase": "draft60", "id": key, "output_tokens": offset})
        print(f"[long60 round] {key} output={offset}/{c['max_new_tokens']} drafting 60 MASKs", flush=True)
        draft = asdict(bundle.drafter.draft(prefix, 60, anchors={}))
        validate_draft(draft, 60, prefix)
        proposal = until_eos(draft["tokens"], bundle.eos)
        store.put("meta", "active", {"phase": "verify60", "id": key})
        feedback = bundle.verifier.score(prefix, proposal)
        decision = decide(proposal, feedback["top1_with_bonus"], bundle.eos, c["max_new_tokens"]-offset)
        row = {"id": key, "prompt_id": prompt["id"], "dataset": prompt["dataset"],
               "round_index": index, "offset": offset, "prefix_digest": digest(prefix),
               "draft": draft, "proposal": proposal, "verification": feedback, "decision": decision}
        store.put("round", key, row)
        emitted.extend(decision["committed_tokens"])
        previous.append(row)
        print(f"[long60 verified] {key} accepted={decision['accepted_draft']}/{len(proposal)} "
              f"emitted={len(decision['committed_tokens'])} total={len(emitted)}", flush=True)
        clear_unused()
        checkpoint()
    return emitted, previous


def audit_and_diagnose(prompt, ids, emitted, rounds, c, store, bundle):
    # No reference/gold enters generate_prompt or online verifier decisions.
    key = prompt["id"]
    reference = store.get("reference", key)
    if reference is None:
        n = c["max_new_tokens"]+c["reference_lookahead"]
        if len(ids)+n > c["context_limit"]:
            raise ValueError("Canonical lookahead exceeds context limit")
        print(f"[long60 audit] {key} independent target-only canonical generation, up to {n} tokens", flush=True)
        result = bundle.canonical(ids, n)
        reference = {"id": key, "prompt_ids": ids, "tokens": result["tokens"], "stats": result["stats"]}
        store.put("reference", key, reference)
    gold = reference["tokens"]
    if gold[:len(emitted)] != emitted:
        mismatch = next((i for i, (x, y) in enumerate(zip(emitted, gold)) if x != y), min(len(emitted), len(gold)))
        raise AssertionError(f"Actual online output differs from canonical greedy target at {key}:{mismatch}; no gold substitution")
    for row in rounds:
        proposal = row["proposal"]
        future = gold[row["offset"]:row["offset"]+len(proposal)]
        canonical_a = next((i for i, (x, y) in enumerate(zip(proposal, future)) if x != y), min(len(proposal), len(future)))
        if canonical_a != row["decision"]["accepted_draft"]:
            raise AssertionError(f"Round acceptance differs from canonical target: {row['id']}; raw verification preserved")
        diagnostic = diagnose(row, gold, bundle.eos, c)
        if diagnostic is not None:
            store.put("diagnostic", row["id"], diagnostic)
    store.put("prompt", key, {**prompt, "prompt_ids": ids, "tokens": emitted,
              "rounds": len(rounds), "terminated": bool(emitted and emitted[-1] in bundle.eos),
              "canonical_audit_passed": True})


def pack(output, store=None):
    if store:
        store.backup(output/"checkpoint.sqlite")
        for kind in ("round", "prompt", "reference", "diagnostic"):
            store.export(kind, output/(kind+"s.jsonl"))
    archive, temp = output/"long60_checkpoint.zip", output/"long60_checkpoint.tmp"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file() and p not in (archive, temp) and not p.name.startswith("long60.sqlite"):
                z.write(p, p.relative_to(output).as_posix())
    temp.replace(archive)
    return archive


def restore(root, output, identity):
    database = output/"long60.sqlite"
    if not root:
        return
    if database.exists():
        raise ValueError("Local checkpoint exists; omit resume-root to continue it")
    root = Path(root)
    artifacts = {}
    preserved = ("token_pieces.json", "decoded_sequences.json", "preflight.json", "runtime.json")
    files = list(root.rglob("checkpoint.sqlite")) if root.is_dir() else []
    if len(files) == 1:
        data = files[0].read_bytes()
        artifacts = {name: (files[0].parent/name).read_bytes() for name in preserved if (files[0].parent/name).exists()}
    else:
        archives = [root] if root.is_file() else list(root.rglob("long60_checkpoint.zip"))
        if len(archives) != 1:
            raise ValueError("Resume root must identify one B60 checkpoint, not an older experiment")
        with zipfile.ZipFile(archives[0]) as z:
            data = z.read("checkpoint.sqlite")
            artifacts = {name: z.read(name) for name in preserved if name in z.namelist()}
    temp = output/"restore.tmp.sqlite"
    temp.write_bytes(data)
    conn = sqlite3.connect(str(temp))
    try:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Resume SQLite integrity failed")
        saved = conn.execute("SELECT payload FROM records WHERE kind='meta' AND key='identity'").fetchone()
        if saved is None or digest(json.loads(saved[0])) != digest(identity):
            raise ValueError("Resume configuration/code/model pins differ")
    finally:
        conn.close()
    temp.replace(database)
    for name, content in artifacts.items():
        (output/name).write_bytes(content)


def execute(output, revision, c, resume_root=None):
    validate_config(c)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT/"source-lock.json").read_text())
    identity = {"protocol": "online-dllm60-ar-shift-diagnostics-v1", "code_revision": revision,
                "source_digest": source_digest(), "config": c,
                "models": {k: lock["models"][k] for k in (TARGET, DLLM)}, "datasets": lock["datasets"]}
    restore(resume_root, output, identity)
    store = Store(output/"long60.sqlite", identity)
    atomic_json(output/"long60-manifest.json", identity)
    started = time.monotonic()
    last_zip = started
    status, bundle, report_error = "failed", None, None
    def boundary():
        if time.monotonic()-started >= c["soft_budget_seconds"]:
            raise BudgetReached()
    def checkpoint(force=False):
        nonlocal last_zip
        if force or time.monotonic()-last_zip >= c["zip_interval_seconds"]:
            pack(output, store)
            last_zip = time.monotonic()
    try:
        plan = store.get("meta", "prompt_plan")
        if plan is None:
            plan = prepare_prompts(c)
            store.put("meta", "prompt_plan", plan)
        if len(plan) != c["prompts_per_dataset"]*len(c["datasets"]) or len({p["id"] for p in plan}) != len(plan):
            raise ValueError("Frozen prompt plan is incomplete or duplicated")
        atomic_json(output/"prompt_plan.json", plan)
        checkpoint(True)
        if not store.get("meta", "complete"):
            bundle = load_bundle(c, output)
            for i, prompt in enumerate(plan):
                if store.get("prompt", prompt["id"]):
                    continue
                boundary()
                print(f"[long60 prompt] {i+1}/{len(plan)} {prompt['id']}", flush=True)
                ids = bundle.tokenizer.apply_chat_template([{"role": "user", "content": prompt["question"]}],
                                                           tokenize=True, add_generation_prompt=True)
                emitted, rounds = generate_prompt(prompt, ids, c, store, bundle, boundary, checkpoint)
                boundary()
                audit_and_diagnose(prompt, ids, emitted, rounds, c, store, bundle)
                from .long60_report import report
                report(output, store, c, bundle.tokenizer)
                clear_unused()
                checkpoint(True)
            store.put("meta", "complete", True)
            store.put("meta", "active", None)
        status = "complete"
    except BudgetReached:
        status = "budget_checkpoint"
        print("[long60] Soft wall budget reached; completed rounds are resumable", flush=True)
    except BaseException as error:
        atomic_json(output/"failure.json", {"type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(), "active": store.get("meta", "active"),
                    "evidence": getattr(error, "evidence", None)})
        raise
    finally:
        try:
            from .long60_report import report
            report(output, store, c, bundle.tokenizer if bundle else None)
        except Exception as error:
            atomic_json(output/"report_failure.json", {"message": str(error)})
            report_error = error
            status = "failed"
        atomic_json(output/"execution_status.json", {"status": status, "code_revision": revision,
                    "completed_prompts": len(store.rows("prompt")), "expected_prompts": c["prompts_per_dataset"]*len(c["datasets"]),
                    "saved_rounds": len(store.rows("round")), "labeled_rejections": len(store.rows("diagnostic")),
                    "operational_seconds_this_session": time.monotonic()-started,
                    "latency_study": False, "operational_wall_includes_peer_transfers": True})
        archive = pack(output, store)
        print(f"[{status}] ZIP: {archive}", flush=True)
        store.close()
        bundle = None
        clear_unused()
    if report_error is not None:
        raise RuntimeError("B60 report export failed; raw checkpoint is preserved") from report_error
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--resume-root")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    c = json.loads((ROOT/"configs/long60.json").read_text())
    c.update(prompts_per_dataset=args.count, max_new_tokens=args.max_new_tokens)
    execute(args.output, args.revision, c, args.resume_root)


if __name__ == "__main__":
    main()
