"""Batch-one inference. No mutable-block KV reuse and no CPU offload."""
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from .assets import TARGET, AR, DLLM, source_digest
from .core import Draft, prediction_row, strict_top1


def release(model):
    import torch
    del model
    gc.collect()
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


def sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def sync_model(model):
    """Account for every GPU in a sharded verifier, including transfer completion."""
    mapping = getattr(model, "hf_device_map", None)
    devices = set(mapping.values()) if mapping else {model.device}
    for device in sorted(devices, key=str):
        sync(f"cuda:{device}" if isinstance(device, int) else device)


def memory():
    import torch
    return [{"device": i, "name": torch.cuda.get_device_name(i),
             "allocated_gib": torch.cuda.memory_allocated(i)/2**30,
             "reserved_gib": torch.cuda.memory_reserved(i)/2**30,
             "total_gib": torch.cuda.get_device_properties(i).total_memory/2**30}
            for i in range(torch.cuda.device_count())]


def configure_numerics(config):
    """Use identical process-wide math settings in preflight and experiment."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config.get(
        "allow_fp16_reduced_precision_reduction", False)


def load_model(path, device, precision="fp16", *, attention="sdpa", device_map=None):
    import torch
    from transformers import AutoModelForCausalLM
    if precision != "fp16":
        raise ValueError("This experiment requires fixed FP16; quantized fallback is disabled")
    options = dict(trust_remote_code=True, local_files_only=True,
                   torch_dtype=torch.float16, device_map=device_map if device_map is not None else {"": device},
                   attn_implementation=attention)
    model = AutoModelForCausalLM.from_pretrained(path, **options).eval()
    if any(str(v) in ("cpu", "disk") for v in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("CPU/disk offload is forbidden in this protocol")
    return model


def load_target(path, config, required=None):
    from transformers import AutoConfig
    if config["target_precision"] != "fp16" or required not in (None, "fp16"):
        raise ValueError("Use a fresh FP16 run; INT8/NF4 references cannot be resumed as FP16")
    architecture = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    split = config["target_gpu0_layers"]
    layers = architecture.num_hidden_layers
    if architecture.model_type != "qwen2" or not 0 < split < layers:
        raise ValueError("Explicit Qwen2 verifier map requires a split inside its decoder layers")
    # Contiguous blocks minimize inter-GPU transfers. GPU1 retains room for one
    # 1.5B FP16 drafter; dLLM and AR are never loaded together.
    mapping = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    mapping.update({f"model.layers.{i}": 0 if i < split else 1 for i in range(layers)})
    model = load_model(path, "cuda:0", "fp16", attention=config.get("target_attention", "sdpa"),
                       device_map=mapping)
    model._suffix_exclude_peer_transfers = config.get("exclude_inter_gpu_transfer_time", True)
    return model, "fp16", None


def tokenizers(paths):
    from transformers import AutoTokenizer
    tokenizers = {name: AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                    local_files_only=True) for name, path in paths.items()}
    base = tokenizers[TARGET]
    vocab = base.get_vocab()
    for name, tokenizer in tokenizers.items():
        # The diffusion checkpoint adds MASK; shared token strings must keep IDs.
        other = tokenizer.get_vocab()
        if any(token in other and other[token] != idx for token, idx in vocab.items()):
            raise AssertionError(f"Incompatible token IDs: {name}")
        if any(token not in other for token in vocab):
            raise AssertionError(f"Target vocabulary absent in {name}")
        for text in ("Solve: 12 + 23 =", "你好 tiếng Việt", "\\boxed{42}", "\n    return x"):
            if base.encode(text) != tokenizer.encode(text):
                raise AssertionError(f"Tokenizer encoding differs: {name}")
    evidence = {"shared_vocab_sha256": hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest(),
                "eos_token_id": base.eos_token_id, "chat_template": base.chat_template}
    return base, evidence


class ARSession:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.timer = None
        if getattr(model, "_suffix_exclude_peer_transfers", False):
            from .timing import StageTimer
            if not hasattr(model, "_suffix_stage_timer"):
                model._suffix_stage_timer = StageTimer(model)
            self.timer = model._suffix_stage_timer

    def draft(self, prefix, length, stop_eos=None, *, anchors=None):
        if self.timer is None:
            result = self._draft(prefix, length, stop_eos, anchors)
            result.stats["latency_policy"] = "single_device_synchronized_wall_no_peer_transfer"
            return result
        with self.timer.measure():
            result = self._draft(prefix, length, stop_eos, anchors)
        if length:
            result.stats.update(self.timer.report(result.stats["latency_ms"]))
        return result

    def _draft(self, prefix, length, stop_eos=None, anchors=None):
        import torch
        anchors = anchors or {}
        if anchors and stop_eos is not None:
            raise ValueError("Constrained AR requires fixed horizon without EOS stopping")
        if any(not 0 <= i < length or token < 0 for i, token in anchors.items()):
            raise ValueError("Invalid AR anchor")
        sync_model(self.model)
        begin = time.perf_counter()
        result, confidence, order = [], [], []
        prefill_ms, mutable_steps = 0., 0
        cache = None
        ids = torch.tensor([prefix], device=self.device)
        input_tokens = 0
        with torch.inference_mode():
            for step in range(length):
                input_tokens += ids.shape[1]
                prefill_start = time.perf_counter() if step == 0 else None
                out = self.model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                if step == 0:
                    sync_model(self.model)
                    prefill_ms = (time.perf_counter()-prefill_start)*1000
                cache = out.past_key_values
                logits = out.logits[0, -1].float()
                token = anchors[step] if step in anchors else int(logits.argmax(-1))
                prob = logits.softmax(-1)[token]
                token = int(token)
                result.append(token)
                confidence.append(float(prob))
                order.append(None if step in anchors else mutable_steps)
                mutable_steps += step not in anchors
                if stop_eos is not None and token in stop_eos:
                    break
                ids = torch.tensor([[token]], device=self.device)
        sync_model(self.model)
        latency = (time.perf_counter()-begin)*1000
        return Draft(result, confidence, order, stats={
            "total_forwards": len(result), "input_tokens": input_tokens,
            "unmask_steps": mutable_steps, "hard_keep_count": len(anchors),
            "forced_positions": len(anchors), "prefill_forwards": int(bool(result)),
            "decode_forwards": max(0, len(result)-1), "prefill_ms": prefill_ms,
            "mutable_latency_ms": max(0., latency-prefill_ms), "latency_ms": latency,
            "mutable_time_definition": "AR wall after causal prefix prefill; forced tokens still require forward processing"})

    def score(self, prefix, proposal, top_k=8):
        """Old causal feedback on draft contexts; never a post-correction acceptance."""
        if self.timer is None:
            return self._score(prefix, proposal, top_k)
        with self.timer.measure():
            result = self._score(prefix, proposal, top_k)
        result.update(self.timer.report(result["latency_ms"]))
        return result

    def _score(self, prefix, proposal, top_k=8):
        import torch
        sync_model(self.model)
        begin = time.perf_counter()
        with torch.inference_mode():
            ids = torch.tensor([prefix + proposal], device=self.device)
            logits = self.model(ids, use_cache=False, logits_to_keep=len(proposal)+1).logits[0, :-1].float()
            if logits.shape[0] != len(proposal):
                raise AssertionError("Shifted verifier row count differs")
            probs = logits.softmax(-1)
            if not 2 <= top_k <= logits.shape[-1]:
                raise ValueError("top_k must be between two and the vocabulary size")
            # topk is not stable for tied logits. Its first ID must never define
            # greedy acceptance: use exactly the argmax rule used in draft().
            winner = logits.argmax(-1, keepdim=True)
            remaining = logits.clone().scatter_(1, winner, -torch.inf)
            others = remaining.topk(top_k-1, -1).indices
            tok = torch.cat((winner, others), dim=1)
            p = probs.gather(1, tok)
            draft_p = probs.gather(1, torch.tensor(proposal, device=logits.device)[:, None]).squeeze(1)
            entropy = -(probs * probs.clamp_min(1e-38).log()).sum(-1)
            answer = {"top1": tok[:, 0].tolist(), "probability": draft_p.tolist(),
                      "top1_probability": p[:, 0].tolist(), "margin": (p[:, 0]-p[:, 1]).tolist(),
                      "logit_margin": (logits.gather(1, winner).squeeze(1)-
                                       logits.gather(1, others[:, :1]).squeeze(1)).tolist(),
                      "entropy": entropy.tolist(), "topk_ids": tok.tolist(), "topk_probs": p.tolist()}
        sync_model(self.model)
        answer["latency_ms"] = (time.perf_counter()-begin)*1000
        answer["input_tokens"] = len(prefix) + len(proposal)
        return answer


class TorchBlockProvider:
    """Only completed physical blocks enter KV; each arm starts a new provider.

    Boundary prediction comes from the previous completed block's last row.
    It competes with other MASKs in its subblock, without a forced seed commit.
    """
    def __init__(self, model, block_size=32, mask_id=151665, use_prefix_cache=True):
        self.model, self.device = model, model.device
        self.block_size, self.mask_id = block_size, mask_id
        self.use_prefix_cache = use_prefix_cache
        self.cache, self.carry = None, None
        self.cached_length = 0
        self.started = False
        self.counts = dict(total_forwards=0, unmask_forwards=0, prefill_forwards=0,
                           cache_update_forwards=0, input_tokens=0,
                           prefill_ms=0., cache_update_ms=0., unmask_ms=0.)

    def forward(self, ids, update, category, keep):
        import torch
        sync(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(torch.tensor([ids], device=self.device),
                past_key_values=self.cache if self.use_prefix_cache else None,
                use_cache=self.use_prefix_cache, update_past_key_values=update,
                block_size=self.block_size, use_block_cache=False, logits_to_keep=keep)
        sync(self.device)
        self.counts[category+"_ms"] += (time.perf_counter()-start)*1000
        self.counts[category+"_forwards"] += 1
        self.counts["total_forwards"] += 1
        self.counts["input_tokens"] += len(ids)
        if update:
            self.cache = out.past_key_values
            self.cached_length += len(ids)
            if self.cache.get_seq_length() != self.cached_length:
                raise AssertionError("Completed-block cache length mismatch")
        return out.logits[0]

    def predict(self, canvas, block_start, positions):
        import torch
        size = self.block_size
        if self.use_prefix_cache:
            if self.cached_length < block_start:
                # May include blocks consisting entirely of hard KEEP anchors.
                completed = canvas[self.cached_length:block_start]
                if self.mask_id in completed:
                    raise AssertionError("Attempt to cache an incomplete block")
                category = "cache_update" if self.started else "prefill"
                rows = self.forward(completed, True, category, 1)
                self.carry = rows[-1]
            if self.cached_length != block_start:
                raise AssertionError("Cache crossed the mutable block")
            before = self.cache.get_seq_length() if self.cache is not None else 0
            rows = self.forward(canvas[block_start:block_start+size], False, "unmask", size)
            after = self.cache.get_seq_length() if self.cache is not None else 0
            if before != after:
                raise AssertionError("Mutable-block forward modified prefix KV")
            carry = self.carry
        else:
            rows = self.forward(canvas[:block_start+size], False, "unmask", size + int(block_start > 0))
            carry = rows[0] if block_start else None
            rows = rows[-size:]
        self.started = True
        selected = []
        for pos in positions:
            row = prediction_row(pos, block_start)
            if row is None and carry is None:
                raise AssertionError("No predecessor for boundary token")
            selected.append(carry if row is None else rows[row])
        with torch.inference_mode():
            logits = torch.stack(selected).float()
            logits[:, self.mask_id] = -torch.inf  # MASK is not an output token.
            token = logits.argmax(-1)
            probability = logits.softmax(-1).gather(1, token[:, None]).squeeze(1)
            return {p: (int(t), float(c)) for p, t, c in zip(positions, token, probability)}

    def stats(self):
        return dict(self.counts)


class DLLMDrafter:
    def __init__(self, model, config):
        self.model, self.config = model, config

    def draft(self, prefix, length, anchors=None, use_prefix_cache=True):
        config = self.config
        provider = TorchBlockProvider(self.model, config["physical_block_size"],
                                      config["mask_id"], use_prefix_cache)
        sync(self.model.device)
        begin = time.perf_counter()
        result = strict_top1(prefix, length, anchors or {}, provider, config["mask_id"],
                             config["physical_block_size"], config["subblock_size"])
        sync(self.model.device)
        result.stats["latency_ms"] = (time.perf_counter()-begin)*1000
        result.stats["scoring_scheduler_ms"] = max(0., result.stats["latency_ms"] - sum(
            result.stats[key] for key in ("prefill_ms", "cache_update_ms", "unmask_ms")))
        # Cold-prefix replay is separately visible; no simulated rollback cost.
        result.stats["mutable_latency_ms"] = result.stats["latency_ms"] - result.stats["prefill_ms"]
        return result


def target_consistency(session, prefix, length=12):
    """Separate repeatability from shifted parallel verification, retaining evidence."""
    if length < 1:
        raise ValueError("Target consistency audit needs at least one token")
    a = session.draft(prefix, length).tokens
    b = session.draft(prefix, length).tokens
    feedback = session.score(prefix, a)
    parallel = feedback["top1"]
    # The last input proposal token is a strict causal future for every returned
    # prediction row. Changing it must not change those rows' greedy predictions.
    alternate = list(a)
    alternate[-1] = next((t for t in prefix if t != a[-1]), 0 if a[-1] != 0 else 1)
    tail_changed = session.score(prefix, alternate)["top1"]
    repeatable, shifted_equal = a == b, a == parallel
    causal_equal = parallel == tail_changed
    mismatches = []
    for i, token in enumerate(a):
        if token == b[i] == parallel[i] == tail_changed[i]:
            continue
        # Full-prefix single-row replay helps distinguish cached decoding from
        # long teacher-forced scoring. It is diagnostic, never substituted as gold.
        single = session.score(prefix+a[:i], [token], top_k=2)
        mismatches.append({"position": i, "cached_token": token, "repeat_token": b[i],
            "parallel_token": parallel[i], "parallel_logit_margin": feedback["logit_margin"][i],
            "tail_changed_token": tail_changed[i],
            "full_prefix_token": single["top1"][0],
            "full_prefix_logit_margin": single["logit_margin"][0]})
        if len(mismatches) == 3:
            break
    return {"passed": repeatable and shifted_equal and causal_equal, "repeatable": repeatable,
            "parallel_shifted_equal": shifted_equal, "greedy_tie_break": "lowest_token_id",
            "causal_tail_equal": causal_equal, "tail_changed_tokens": tail_changed,
            "prefix_length": len(prefix), "cached_tokens": a, "repeat_tokens": b,
            "parallel_tokens": parallel, "first_mismatches": mismatches}


def constrained_ar_consistency(session, prefix, length=8):
    """Check actual forced KV propagation against independent full-prefix replay."""
    import torch
    fresh = session.draft(prefix, length)
    anchors = {length//2: prefix[-1], length-1: prefix[0]}
    actual = session.draft(prefix, length, anchors=anchors)
    expected = []
    with torch.inference_mode():
        for i in range(length):
            ids = torch.tensor([prefix+expected], device=session.device)
            logits = session.model(ids, use_cache=False, logits_to_keep=1).logits[0, -1]
            expected.append(anchors[i] if i in anchors else int(logits.argmax()))
    all_keep = session.draft(prefix, length, anchors=dict(enumerate(fresh.tokens)))
    report = {"forced_full_prefix_equal": actual.tokens == expected,
        "future_anchor_causal_equal": actual.tokens[:length//2] == fresh.tokens[:length//2],
        "all_keep_equal": all_keep.tokens == fresh.tokens,
        "all_keep_forwards": all_keep.stats["total_forwards"],
        "forced_tokens": [[i, t] for i, t in sorted(anchors.items())]}
    report["passed"] = all(report[k] for k in ("forced_full_prefix_equal", "future_anchor_causal_equal", "all_keep_equal")) and report["all_keep_forwards"] == length
    return report


class TargetConsistencyError(AssertionError):
    def __init__(self, attempts):
        self.attempts = attempts
        super().__init__("No verifier configuration passed exact target consistency")


def checked_target(path, config, prefix, required=None):
    """Check the fixed FP16 verifier; never switch precision to bypass a failure."""
    model, precision, fallback = load_target(path, config, required)
    session = ARSession(model)
    audit = target_consistency(session, prefix)
    attempts = [{"precision": precision, "load_fallback": fallback, "target_audit": audit}]
    if audit["passed"]:
        return model, precision, fallback, audit, attempts
    del session, model
    release(None)
    raise TargetConsistencyError(attempts)


def preflight(manifest, config, output):
    """Real pretrained GPU checks; success is required before the experiment."""
    import torch
    from .store import atomic_json
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Requires two CUDA GPUs (Kaggle T4 x2)")
    torch.manual_seed(config["seed"])
    configure_numerics(config)
    tokenizer, evidence = tokenizers(manifest["paths"])
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": "Compute 17 + 28. Explain briefly."}],
                                           tokenize=True, add_generation_prompt=True)
    established = Path(output) / "run-manifest.json"
    previous = json.loads(established.read_text()) if established.exists() else None
    if previous and (previous["config"] != config or previous["asset_digest"] != manifest["content_digest"]):
        raise ValueError("Existing run differs; preflight requires its original config/assets or a new output")
    try:
        target, precision, fallback, audit, attempts = checked_target(manifest["paths"][TARGET], config,
                                                 prefix, previous["target_precision"] if previous else None)
    except TargetConsistencyError as error:
        last = error.attempts[-1]
        precision, fallback, audit = last["precision"], last["load_fallback"], last["target_audit"]
        failure = {"passed": False, "stage": "target_consistency", "precision": precision,
                   "fallback": fallback, "target_audit": audit, "torch": torch.__version__,
                   "target_attempts": error.attempts,
                   "config": config, "assets_digest": manifest["content_digest"]}
        failure["source_digest"] = source_digest()
        atomic_json(Path(output)/"preflight_failure.json", failure)
        atomic_json(Path(output)/"preflight.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise AssertionError("Target consistency failed; see preflight_failure.json for repeatability, "
                             "parallel tokens and first mismatch margins. No experiment was run.")
    target_memory = memory()
    target_device_map = {name: str(device) for name, device in target.hf_device_map.items()}
    # Keep the FP16 verifier resident while checking one drafter at a time.
    dllm = load_model(manifest["paths"][DLLM], "cuda:1")
    drafter = DLLMDrafter(dllm, config)
    checks = []
    for phase in (0, 1, 7, 8, 31):
        # Extend with known prompt tokens solely for geometry checks.
        p = prefix + [prefix[-1]] * ((phase-len(prefix)) % 32)
        gold = drafter.draft(p, 10)
        replay = drafter.draft(p, 10, use_prefix_cache=False)
        if gold.tokens != replay.tokens:
            raise AssertionError(f"Cached/full block predictions differ at phase {phase}")
        anchors = {i: t for i, t in enumerate(gold.tokens) if i % 2 == 0}
        x = drafter.draft(p, 10, anchors)
        y = drafter.draft(p, 10, anchors)
        if x.tokens != y.tokens or x.stats["unmask_steps"] != 10-len(anchors):
            raise AssertionError("Independent KEEP branches failed")
        checks.append({"phase": phase, "cached_full_equal": True, "keep_independent": True})
    dllm_memory = memory()
    del drafter, dllm
    release(None)
    ar = load_model(manifest["paths"][AR], "cuda:1")
    ar_audit = target_consistency(ARSession(ar), prefix, 8)
    constrained_ar_audit = constrained_ar_consistency(ARSession(ar), prefix, 8)
    if not ar_audit["passed"] or not constrained_ar_audit["passed"]:
        raise AssertionError("FP16 AR drafter preflight consistency failed")
    ar_memory = memory()
    del ar, target
    release(None)
    report = {"passed": True, "precision": precision, "fallback": fallback,
              "assets_digest": manifest["content_digest"], "config": config,
              "target_audit": audit,
              "target_attempts": attempts,
              "target_device_map": target_device_map,
              "target_latency_policy": "cuda_stage_events_excluding_dispatch_copies",
              "ar_audit": ar_audit, "constrained_ar_audit": constrained_ar_audit, "ar_memory": ar_memory,
              "tokenizer": evidence, "checks": checks, "target_memory": target_memory,
              "dllm_memory": dllm_memory, "torch": torch.__version__}
    report["source_digest"] = source_digest()
    atomic_json(Path(output) / "preflight.json", report)
    (Path(output)/"preflight_failure.json").unlink(missing_ok=True)
    return report
