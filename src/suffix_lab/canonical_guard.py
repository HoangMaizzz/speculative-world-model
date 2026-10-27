"""Online numerical guard using actual target KV predictions, never reference labels.

The extra serial target work is explicit: this is audited SD for data collection,
not a one-target-forward-per-round speed benchmark.
"""
from .store import digest


class CanonicalCursor:
    def __init__(self, model, eos):
        self.model, self.eos = model, set(eos)
        self.cache = self.logits = None
        self.prefix = []
        self.forwards = self.input_tokens = 0
        self.terminal = False

    def _forward(self, ids):
        import torch
        with torch.inference_mode():
            out = self.model(torch.tensor([ids], device=self.model.device), past_key_values=self.cache,
                             use_cache=True, logits_to_keep=1)
        self.cache, self.logits = out.past_key_values, out.logits[0, -1].float()
        self.forwards += 1
        self.input_tokens += len(ids)

    def begin(self, prompt, committed):
        self.cache = self.logits = None
        self.prefix = list(prompt)
        self.forwards = self.input_tokens = 0
        self.terminal = False
        self._forward(list(prompt))
        for i, token in enumerate(committed):
            actual, margin = self.prediction()
            if token != actual:
                error = AssertionError(f"Saved committed prefix is not canonical at output position {i}")
                error.evidence = {"position": i, "saved_token": token, "actual_token": actual,
                                  "actual_logit_margin": margin}
                raise error
            self.advance(token)

    def prediction(self):
        if self.terminal:
            raise ValueError("No target predictions after committed EOS")
        values = self.logits.topk(2).values
        return int(self.logits.argmax()), float(values[0]-values[1])

    def advance(self, token):
        self.prefix.append(int(token))
        if token in self.eos:
            self.terminal = True
        else:
            self._forward([int(token)])


def guard_feedback(parallel, proposal, cursor, eos):
    original = parallel["top1_with_bonus"]
    if len(original) != len(proposal)+1:
        raise ValueError("Parallel rows/bonus missing")
    authoritative = list(original)
    margins = list(parallel.get("logit_margin", [None]*len(original)))
    before, tokens_before = cursor.forwards, cursor.input_tokens
    checked, differences = 0, []
    for i in range(len(proposal)+1):
        actual, margin = cursor.prediction()
        authoritative[i], margins[i] = actual, margin
        checked += 1
        if actual != original[i]:
            differences.append({"position": i, "parallel_token": original[i], "canonical_token": actual,
                                "parallel_logit_margin": parallel.get("logit_margin", [None]*len(original))[i],
                                "canonical_logit_margin": margin})
        # Advance only true canonical tokens, never an unchecked draft token.
        cursor.advance(actual)
        if actual in eos or i == len(proposal) or actual != proposal[i]:
            break
    guard_forwards = cursor.forwards-before
    return {"top1_with_bonus": authoritative, "logit_margin": margins,
            "verification_mode": "online_canonical_guarded_parallel",
            "parallel_feedback": parallel,
            "canonical_guard": {"checked_rows": checked, "new_forwards": guard_forwards,
                                "new_input_tokens": cursor.input_tokens-tokens_before,
                                "numerical_disagreements": differences,
                                "unchecked_tail_rows_are_parallel_only": True},
            "forwards": parallel.get("forwards", 1)+guard_forwards,
            "input_tokens": parallel.get("input_tokens", 0)+cursor.input_tokens-tokens_before}


class GuardedVerifier:
    def __init__(self, parallel, eos):
        self.parallel = parallel
        self.eos = set(eos)
        self.cursor = CanonicalCursor(parallel.model, eos)
        self.replay_stats = None

    def begin_prompt(self, ids, committed):
        self.cursor.begin(ids, committed)
        self.replay_stats = {"forwards": self.cursor.forwards, "input_tokens": self.cursor.input_tokens,
                             "committed_tokens_replayed": len(committed),
                             "prefix_digest": digest(ids+committed)}

    def score(self, prefix, proposal):
        if self.cursor.prefix != list(prefix):
            raise ValueError("Canonical verifier KV prefix differs from actual committed prefix")
        parallel = self.parallel.score(prefix, proposal)
        result = guard_feedback(parallel, proposal, self.cursor, self.eos)
        if self.replay_stats is not None:
            result["canonical_prefix_replay"] = self.replay_stats
            result["forwards"] += self.replay_stats["forwards"]
            result["input_tokens"] += self.replay_stats["input_tokens"]
            self.replay_stats = None
        return result
