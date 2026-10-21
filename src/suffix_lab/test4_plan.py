"""Outcome-blind Main200 sampling, EOS semantics and immutable action identities."""
from collections import Counter
import random
import unicodedata
from .metrics import prefix_length
from .store import digest


def through_eos(tokens, eos_ids):
    stop = next((i+1 for i, t in enumerate(tokens) if t in eos_ids), len(tokens))
    return list(tokens[:stop])


def token_type(piece, token, special_ids):
    if token in special_ids:
        return "special_control"
    if any(c in piece for c in "\r\n\t"):
        return "newline_indentation"
    if piece and piece.isspace():
        return "whitespace_only"
    visible = [c for c in piece if not c.isspace()]
    if visible and all(unicodedata.category(c).startswith("P") for c in visible):
        return "punctuation"
    if visible and all(c.isnumeric() for c in visible):
        return "numeric"
    return "lexical_mixed"


def prompt_plan(primary, datasets, config):
    """Original pilot100 remain primary; all other test prompts are seeded reserves."""
    plan = {}
    for ds in config["datasets"]:
        initial = [dict(p) for p in primary if p["dataset"] == ds]
        ids = {p["id"] for p in initial}
        if len(initial) != config["events_per_dataset"] or len(ids) != len(initial):
            raise ValueError("Expected distinct original primary prompts per benchmark")
        candidates = datasets[ds]
        for p in initial:
            if candidates[p["index"]]["question"] != p["question"]:
                raise ValueError("Pinned dataset and pilot question differ")
        reserve = [dict(p) for p in candidates if p["id"] not in ids]
        random.Random(digest([config["seed"], ds, "reserve"])).shuffle(reserve)
        plan[ds] = [{**p, "source": "primary" if i < len(initial) else "reserve",
                     "offsets": list(config["offsets"])} for i, p in enumerate(initial+reserve)]
    return plan


def choose_event(events, prompt_id, seed):
    """Uniform choice among ALL eligible fixed offsets; no stateful RNG on resume."""
    if not events:
        return None
    ordered = sorted(events, key=lambda e: e["offset"])
    return random.Random(digest([seed, prompt_id, "event-choice"])).choice(ordered)


def make_event(prompt, offset, ref, original, feedback, config, eos_ids, model_revisions):
    draft = original["tokens"]
    if len(draft) != 24:
        raise ValueError("Main requires an actual native B24 tensor")
    logical = through_eos(draft, eos_ids)
    if len(logical) < 24:
        return None, "early_draft_eos"
    gold = through_eos(ref["tokens"][offset:offset+24], eos_ids)
    if not gold:
        return None, "offset_after_reference_end"
    a = prefix_length(draft, gold)
    if a == 24:
        return None, "all_24_accepted"
    if a >= len(gold):
        return None, "reference_censored"
    if feedback["top1"][a] != gold[a]:
        raise AssertionError("Actual target correction differs from canonical reference")
    correction = gold[a]
    if correction in eos_ids:
        return None, "terminal_correction"
    start = offset+a+1
    continuation = through_eos(ref["tokens"][start:start+23], eos_ids)
    terminal = bool(continuation and continuation[-1] in eos_ids)
    if len(continuation) < 23 and not terminal:
        return None, "reference_censored"
    old = draft[a+1:]
    correct = [i for i, t in enumerate(old) if i < len(continuation)
               and t not in eos_ids and continuation[i] not in eos_ids and t == continuation[i]]
    prefix = ref["prompt_ids"]+ref["tokens"][:start]
    event = {"event_id": f"{prompt['id']}:{offset}:dllm24", "prompt_id": prompt["id"],
             "dataset": prompt["dataset"], "question": prompt["question"], "offset": offset,
             "accepted_before_rejection": a, "correction_token": correction,
             "reference_start": start, "prefix": prefix, "reference": continuation,
             "old_context": ref["prompt_ids"]+ref["tokens"][:offset],
             "old": old, "L": 23-a, "extension_length": a, "H": 23,
             "original_draft": original, "original_verification": feedback,
             "oracle_anchors": [[i, old[i]] for i in correct], "oracle_keep_positions": correct,
             "old_correct_count": len(correct), "physical_phase": len(prefix)%32,
             "old_endpoint_phase8": (len(prefix)+len(old))%8,
             "target_terminal_within_horizon": terminal,
             "reference_censored": False, "reuse_applicable": bool(old and correct)}
    event["event_fingerprint"] = digest({"prompt_id": prompt["id"], "old_prefix": event["old_context"],
        "original_ids": draft, "corrected_prefix": prefix, "models": model_revisions})
    return event, None


def plans(event, eos_ids):
    # KEEP-All is non-oracle: never inspect the future target's EOS location.
    # Original EOS can only occur in its final slot (earlier ones were excluded).
    keep_all = [[i, t] for i, t in enumerate(event["old"]) if t not in eos_ids]
    return [{"method": name, "anchors": anchors, "horizon": 23, "schedule": "native_joint",
             "oracle": oracle} for name, anchors, oracle in (
                 ("F-23", [], False), ("O-All-Joint-23", event["oracle_anchors"], True),
                 ("KEEP-All-23", keep_all, False))]


def arm_fingerprint(event, plan, config, revision, models):
    return digest({"event": event["event_fingerprint"], "anchors": sorted(plan["anchors"]),
        "horizon": plan["horizon"], "schedule": plan["schedule"], "L": event["L"],
        "decoding_config": config, "code_revision": revision, "models": models})


def eos_metrics(candidate, gold, eos_ids, horizon=23):
    logical = through_eos(candidate, eos_ids)
    target = through_eos(gold, eos_ids)
    n = min(len(logical), len(target), horizon)
    accepted = prefix_length(logical[:n], target[:n])
    terminal = bool(target and target[-1] in eos_ids)
    ended = bool(terminal and accepted == len(target) and n == len(target)
                 and logical[len(target)-1] == target[-1])
    mismatch = accepted if accepted < n else None
    return {"A_inclusive": accepted, "A_content": accepted-int(ended), "I_EOS": int(ended),
            "accepted_prefix": accepted, "terminated_correctly": ended,
            "target_terminal_within_horizon": terminal, "evaluated_tokens": n,
            "logical_proposal_length": len(logical), "logical_tokens": logical,
            "tokens_beyond_logical_eos": len(candidate)-len(logical),
            "tokens_not_evaluated_after_target_end": max(0, len(logical)-len(target)),
            "first_mismatch_position_0based": mismatch,
            "ceiling": accepted == len(target), "full_23_accepted": accepted == horizon}


def changes(event, tokens, fresh, anchors, eos_ids):
    """Compare only mutually valid logical positions, never post-EOS padding."""
    keep = dict(anchors)
    t, f = through_eos(tokens, eos_ids), through_eos(fresh, eos_ids)
    gold = event["reference"]
    limit = min(len(t), len(f), len(gold))
    result = Counter()
    first = min(keep, default=None)
    for i in range(limit):
        if i in keep or gold[i] in eos_ids:
            continue
        label = ("wrong_to_right" if f[i] != gold[i] and t[i] == gold[i] else
                 "right_to_wrong" if f[i] == gold[i] and t[i] != gold[i] else
                 "both_correct" if f[i] == t[i] == gold[i] else "both_wrong")
        result[label] += 1
        result[("REDRAFT_" if i < event["L"] else "EXTEND_")+label] += 1
        if first is not None and i < first:
            result[label+"_before_first_anchor"] += 1
    af = eos_metrics(fresh, gold, eos_ids)["A_inclusive"]
    at = eos_metrics(tokens, gold, eos_ids)["A_inclusive"]
    attribution = Counter()
    for i in range(af, at):
        name = ("terminal_eos_corrected" if gold[i] in eos_ids else
                "direct_keep_corrected" if i in keep and (i >= len(f) or f[i] != gold[i]) else
                "keep_already_correct" if i in keep else
                "bridged_already_correct_nonkeep" if i < len(f) and f[i] == gold[i] else
                "conditioning_redraft_corrected" if i < event["L"] else "conditioning_extension_corrected")
        attribution[name] += 1
    assert sum(attribution.values()) == max(0, at-af)
    jf = eos_metrics(fresh, gold, eos_ids)["first_mismatch_position_0based"]
    return {"mutable_changes": dict(result), "mutable_comparison_positions": limit,
            "accepted_gain_attribution": dict(attribution), "j_F": jf,
            "keeps_fresh_mismatch": jf in keep if jf is not None else False,
            "correct_anchors_before_jF": sum(i < jf and v == gold[i] for i, v in keep.items()
                if jf is not None and i < len(gold)),
            "correct_anchors_after_jF": sum(i > jf and v == gold[i] for i, v in keep.items()
                if jf is not None and i < len(gold))}
