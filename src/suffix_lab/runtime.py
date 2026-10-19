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
    torch.cuda.empty_cache()


def sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def memory():
    import torch
    return [{"device": i, "name": torch.cuda.get_device_name(i),
             "allocated_gib": torch.cuda.memory_allocated(i)/2**30,
             "reserved_gib": torch.cuda.memory_reserved(i)/2**30,
             "total_gib": torch.cuda.get_device_properties(i).total_memory/2**30}
            for i in range(torch.cuda.device_count())]


def load_model(path, device, precision="fp16"):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    options = dict(trust_remote_code=True, local_files_only=True,
                   torch_dtype=torch.float16, device_map={"": device}, attn_implementation="sdpa")
    if precision == "int8":
        options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
    elif precision != "fp16":
        raise ValueError(precision)
    model = AutoModelForCausalLM.from_pretrained(path, **options).eval()
    if any(str(v) in ("cpu", "disk") for v in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("CPU/disk offload is forbidden in this protocol")
    return model


def load_target(path, config, required=None):
    precision = required or config["target_precision"]
    try:
        return load_model(path, "cuda:0", precision), precision, None
    except Exception as error:
        if required or precision != "int8" or not config["allow_nf4_fallback"]:
            raise
        # Never fall back during an established run or after generating references.
        reason = {"type": type(error).__name__, "message": str(error)[:500]}
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        return load_model(path, "cuda:0", "nf4"), "nf4", reason


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

    def draft(self, prefix, length, stop_eos=None):
        import torch
        sync(self.device)
        begin = time.perf_counter()
        result, confidence = [], []
        cache = None
        ids = torch.tensor([prefix], device=self.device)
        input_tokens = 0
        with torch.inference_mode():
            for _ in range(length):
                input_tokens += ids.shape[1]
                out = self.model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                logits = out.logits[0, -1].float()
                token = logits.argmax(-1)  # Lowest token ID wins exact ties.
                prob = logits.softmax(-1)[token]
                token = int(token)
                result.append(token)
                confidence.append(float(prob))
                if stop_eos is not None and token in stop_eos:
                    break
                ids = torch.tensor([[token]], device=self.device)
        sync(self.device)
        return Draft(result, confidence, list(range(len(result))), stats={
            "total_forwards": len(result), "input_tokens": input_tokens,
            "latency_ms": (time.perf_counter()-begin)*1000})

    def score(self, prefix, proposal, top_k=8):
        """Old causal feedback on draft contexts; never a post-correction acceptance."""
        import torch
        sync(self.device)
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
            draft_p = probs.gather(1, torch.tensor(proposal, device=self.device)[:, None]).squeeze(1)
            entropy = -(probs * probs.clamp_min(1e-38).log()).sum(-1)
            answer = {"top1": tok[:, 0].tolist(), "probability": draft_p.tolist(),
                      "top1_probability": p[:, 0].tolist(), "margin": (p[:, 0]-p[:, 1]).tolist(),
                      "logit_margin": (logits.gather(1, winner).squeeze(1)-
                                       logits.gather(1, others[:, :1]).squeeze(1)).tolist(),
                      "entropy": entropy.tolist(), "topk_ids": tok.tolist(), "topk_probs": p.tolist()}
        sync(self.device)
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
    a = session.draft(prefix, length).tokens
    b = session.draft(prefix, length).tokens
    feedback = session.score(prefix, a)
    parallel = feedback["top1"]
    repeatable, shifted_equal = a == b, a == parallel
    mismatches = []
    for i, token in enumerate(a):
        if token == b[i] == parallel[i]:
            continue
        # Full-prefix single-row replay helps distinguish cached decoding from
        # long teacher-forced scoring. It is diagnostic, never substituted as gold.
        single = session.score(prefix+a[:i], [token], top_k=2)
        mismatches.append({"position": i, "cached_token": token, "repeat_token": b[i],
            "parallel_token": parallel[i], "parallel_logit_margin": feedback["logit_margin"][i],
            "full_prefix_token": single["top1"][0],
            "full_prefix_logit_margin": single["logit_margin"][0]})
        if len(mismatches) == 3:
            break
    return {"passed": repeatable and shifted_equal, "repeatable": repeatable,
            "parallel_shifted_equal": shifted_equal, "greedy_tie_break": "lowest_token_id",
            "prefix_length": len(prefix), "cached_tokens": a, "repeat_tokens": b,
            "parallel_tokens": parallel, "first_mismatches": mismatches}


def preflight(manifest, config, output):
    """Real pretrained GPU checks; success is required before the experiment."""
    import torch
    from .store import atomic_json
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Requires two CUDA GPUs (Kaggle T4 x2)")
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer, evidence = tokenizers(manifest["paths"])
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": "Compute 17 + 28. Explain briefly."}],
                                           tokenize=True, add_generation_prompt=True)
    established = Path(output) / "run-manifest.json"
    previous = json.loads(established.read_text()) if established.exists() else None
    if previous and (previous["config"] != config or previous["asset_digest"] != manifest["content_digest"]):
        raise ValueError("Existing run differs; preflight requires its original config/assets or a new output")
    target, precision, fallback = load_target(manifest["paths"][TARGET], config,
                                             previous["target_precision"] if previous else None)
    target_session = ARSession(target)
    audit = target_consistency(target_session, prefix)
    if not audit["passed"]:
        failure = {"passed": False, "stage": "target_consistency", "precision": precision,
                   "fallback": fallback, "target_audit": audit, "torch": torch.__version__,
                   "config": config, "assets_digest": manifest["content_digest"]}
        failure["source_digest"] = source_digest()
        atomic_json(Path(output)/"preflight_failure.json", failure)
        atomic_json(Path(output)/"preflight.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise AssertionError("Target consistency failed; see preflight_failure.json for repeatability, "
                             "parallel tokens and first mismatch margins. No experiment was run.")
    target_memory = memory()
    del target_session, target
    release(None)
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
    report = {"passed": True, "precision": precision, "fallback": fallback,
              "assets_digest": manifest["content_digest"], "config": config,
              "target_audit": audit,
              "tokenizer": evidence, "checks": checks, "target_memory": target_memory,
              "dllm_memory": dllm_memory, "torch": torch.__version__}
    report["source_digest"] = source_digest()
    atomic_json(Path(output) / "preflight.json", report)
    return report
