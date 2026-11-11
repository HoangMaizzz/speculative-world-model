"""Run a paired, oracle-placement Fast-dLLM/Qwen repair pilot from a frozen plan."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean
import time
from zipfile import ZipFile, ZIP_DEFLATED

from suffix_lab.long60_plan import decide, until_eos
from suffix_lab.store import Store, atomic_json


ROOT = Path(__file__).resolve().parents[1]
PLAN_MEMBER = "soft_anchor_plan.json"
PLAN_SHA256 = "470499c62213b56a427dcbb51041c7368c21286d487abba7885f650d2fce7acf"


def read_plan(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            raw = archive.read(PLAN_MEMBER)
    else:
        raw = path.read_bytes()
    if sha256(raw).hexdigest() != PLAN_SHA256:
        raise ValueError("Frozen plan SHA-256 differs. Use the published pilot input ZIP.")
    plan = json.loads(raw)
    if (plan["protocol"], plan["length"], plan["block_size"], plan["subblock_size"]) != ("soft-anchor-pilot-v1", 64, 32, 8):
        raise ValueError("Plan geometry differs")
    if len({case["id"] for case in plan["cases"]}) != len(plan["cases"]):
        raise ValueError("Duplicate case in frozen plan")
    for case in plan["cases"]:
        if not case["prompt"] or not case["gold"] or case["remaining"] != 256 - len(case["committed"]):
            raise ValueError("Incomplete corrected context or output budget")
        if any(token in plan["eos"] for token in case["committed"]):
            raise ValueError("Committed EOS before the test round")
        anchor = case["anchor"]
        if anchor is None:
            if case["cohort"] != "representative":
                raise ValueError("Enriched case has no anchor")
            continue
        start, tokens = anchor["destination"], anchor["tokens"]
        length, end = len(tokens), start + len(tokens)
        phase = (len(case["prompt"]) + len(case["committed"])) % 32
        if not (4 <= length <= 8 and 1 <= start <= 7 and start != anchor["source_start"]
                and phase + end <= 32 and end <= min(len(case["gold"]), case["remaining"])):
            raise ValueError("Invalid shifted anchor geometry")
        errors = [i for i, token in enumerate(tokens) if token != case["gold"][start + i]]
        if errors != anchor["errors"] or sorted(anchor["rank"]) != list(range(length)):
            raise ValueError("Oracle mask or source score order differs")
        expected = {"exact": 0, "one_error": 1, "two_errors": 2}.get(case["cohort"])
        if expected is not None and len(errors) != expected:
            raise ValueError("Enriched cohort label differs")
        if case["cohort"] == "two_errors" and length < 6:
            raise ValueError("Two-mask arm requires at least four retained tokens")
    return plan


def actions(case):
    result = {"fresh64": []}
    anchor = case["anchor"]
    if anchor is None:
        return result
    rng = random.Random("soft-anchor-mask-v1:" + case["id"])
    order = rng.sample(range(len(anchor["tokens"])), len(anchor["tokens"]))
    result.update(hard=[], random1=order[:1], head1=anchor["rank"][:1])
    if len(anchor["errors"]) == 1:
        result["oracle1"] = anchor["errors"]
    if case["cohort"] == "two_errors":
        result.update(random2=order[:2], head2=anchor["rank"][:2], oracle2=anchor["errors"])
    return result


def keep_map(case, arm, masks):
    if arm == "fresh64":
        return {}
    anchor = case["anchor"]
    return {anchor["destination"] + i: token for i, token in enumerate(anchor["tokens"]) if i not in masks}


def measures(case, arm, masks, proposal, decision):
    accepted = min(decision["accepted_draft"], case["remaining"])
    out = dict(accepted=accepted, committed=len(decision["committed_tokens"]))
    anchor = case["anchor"]
    if anchor is None:
        return out
    start, end = anchor["destination"], anchor["destination"] + len(anchor["tokens"])
    out.update(bridge_exact=len(proposal) >= start and proposal[:start] == case["gold"][:start],
               reached_anchor=accepted >= start, whole_anchor_accepted=accepted >= end,
               source_equal_and_accepted=sum(start + i < accepted and proposal[start + i] == token
                                             for i, token in enumerate(anchor["tokens"])),
               explicitly_kept_accepted=sum(i < accepted for i in keep_map(case, arm, masks)))
    if masks:
        correct = [start + i < len(proposal) and proposal[start + i] == case["gold"][start + i] for i in masks]
        out.update(mask_positions=len(masks), mask_correct=sum(correct), all_masks_correct=all(correct),
                   correct_tokens_remasked=sum(i not in anchor["errors"] for i in masks),
                   correct_tokens_damaged=sum(i not in anchor["errors"] and not ok
                                              for i, ok in zip(masks, correct, strict=True)))
    return out


class ReferenceMismatch(AssertionError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__("Actual guarded Qwen disagrees with the saved reference")


def execute_arm(case, arm, masks, bundle):
    from suffix_lab.runtime import sync_model
    prefix = case["prompt"] + case["committed"]
    keep = keep_map(case, arm, masks)
    draft = bundle.drafter.draft(prefix, 64, anchors=keep)
    if (len(draft.tokens) != 64 or draft.stats["unmask_forwards"] != 64 - len(keep)
            or any(draft.tokens[i] != token for i, token in keep.items())):
        raise AssertionError("Strict one-MASK draft or KEEP contract failed")
    proposal = until_eos(draft.tokens, bundle.eos)
    verifier = bundle.verifier
    sync_model(verifier.parallel.model)
    started = time.perf_counter()
    try:
        verifier.begin_prompt(case["prompt"], case["committed"])
    except AssertionError as error:
        if not hasattr(error, "evidence"):
            raise
        raise ReferenceMismatch({"stage": "prefix_replay", **error.evidence}) from error
    sync_model(verifier.parallel.model)
    prefix_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    feedback = verifier.score(prefix, proposal)
    sync_model(verifier.parallel.model)
    verify_ms = (time.perf_counter() - started) * 1000
    decision = decide(proposal, feedback["top1_with_bonus"], bundle.eos, case["remaining"])
    committed = decision["committed_tokens"]
    if committed != case["gold"][:len(committed)]:
        raise ReferenceMismatch(dict(stage="commit", actual=committed, expected=case["gold"][:len(committed)],
                                     proposal=proposal, feedback=feedback, decision=decision))
    return dict(arm=arm, masks=masks, keep=keep, proposal=proposal, decision=decision,
                draft_stats=draft.stats, verification=feedback, prefix_replay_ms=prefix_ms,
                guarded_verify_ms=verify_ms, metrics=measures(case, arm, masks, proposal, decision))


def visibility_probe(cases, drafter):
    """Compare same-shape logits after changing one right-context token."""
    from suffix_lab.runtime import TorchBlockProvider
    for case in cases:
        anchor = case["anchor"]
        if case["cohort"] != "one_error" or anchor["errors"][0] == len(anchor["tokens"]) - 1:
            continue
        hole = anchor["errors"][0]
        prefix = case["prompt"] + case["committed"]
        block = len(prefix) // 32 * 32
        target = len(prefix) + anchor["destination"] + hole
        mask_id = drafter.config["mask_id"]
        provider = TorchBlockProvider(drafter.model, 32, mask_id, False)
        canvas = prefix + [mask_id] * (block + 64 - len(prefix))
        for i, token in keep_map(case, "hard", [hole]).items():
            canvas[len(prefix) + i] = token

        def logits(sequence):
            return provider.forward(sequence, False, "unmask", len(sequence) - target + 1)[0].float().cpu()

        base = logits(canvas)
        same = list(canvas)
        same[target + 1] = mask_id
        across = list(canvas)
        across[block + 32] = canvas[target + 1]
        return dict(case=case["id"], position=target, prediction_row=target - 1,
                    same_block_max_delta=float((base - logits(same)).abs().max()),
                    across_block_max_delta=float((base - logits(across)).abs().max()))
    return {"available": False, "reason": "No one-error probe with right context in this mode"}


def summarize(store):
    groups = defaultdict(list)
    for result in store.rows("case"):
        by_arm = {arm["arm"]: arm for arm in result["arms"]}
        for arm in result["arms"]:
            groups[result["cohort"], arm["arm"]].append((arm, by_arm["fresh64"], by_arm.get("hard")))
    rows = []
    for (cohort, name), pairs in sorted(groups.items()):
        metrics = [a["metrics"] for a, _, _ in pairs]
        delta = [a["accepted"] - fresh["metrics"]["accepted"] for a, (_, fresh, _) in zip(metrics, pairs, strict=True)]
        hard_delta = [arm["metrics"]["accepted"] - hard["metrics"]["accepted"] for arm, _, hard in pairs if hard]
        reached = [row for row in metrics if row.get("reached_anchor")]
        reached_mask = [row for row in reached if "all_masks_correct" in row]
        rows.append(dict(cohort=cohort, arm=name, cases=len(pairs), mean_accepted=mean(r["accepted"] for r in metrics),
                         mean_committed=mean(r["committed"] for r in metrics), mean_delta_vs_fresh=mean(delta),
                         wins_vs_fresh=sum(x > 0 for x in delta), losses_vs_fresh=sum(x < 0 for x in delta),
                         mean_delta_vs_hard=mean(hard_delta) if hard_delta else None,
                         wins_vs_hard=sum(x > 0 for x in hard_delta), losses_vs_hard=sum(x < 0 for x in hard_delta),
                         anchor_cases=sum("reached_anchor" in r for r in metrics), reached_anchor=len(reached),
                         whole_anchor_accepted=sum(r.get("whole_anchor_accepted", False) for r in metrics),
                         mask_correct=sum(r.get("mask_correct", 0) for r in metrics),
                         mask_positions=sum(r.get("mask_positions", 0) for r in metrics),
                         reached_mask_cases=len(reached_mask), reached_all_masks_correct=sum(r["all_masks_correct"] for r in reached_mask),
                         correct_tokens_damaged=sum(r.get("correct_tokens_damaged", 0) for r in metrics),
                         explicitly_kept_accepted=sum(r.get("explicitly_kept_accepted", 0) for r in metrics),
                         mean_draft_ms=mean(a["draft_stats"]["latency_ms"] for a, _, _ in pairs),
                         mean_guarded_verify_ms=mean(a["guarded_verify_ms"] for a, _, _ in pairs),
                         mean_offline_prefix_replay_ms=mean(a["prefix_replay_ms"] for a, _, _ in pairs)))
    return rows


def checkpoint(output, store, status):
    rows = summarize(store)
    atomic_json(output / "summary.json", rows)
    atomic_json(output / "execution_status.json", status)
    if rows:
        temp = output / "summary.csv.tmp"
        with temp.open("w", encoding="utf8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(output / "summary.csv")
    store.export("case", output / "cases.jsonl")
    store.export("failure", output / "numerical_failures.jsonl")
    store.backup(output / "checkpoint.sqlite")
    target = output / "soft_anchor_pilot_results.zip"
    temp = target.with_suffix(".tmp")
    with ZipFile(temp, "w", ZIP_DEFLATED) as archive:
        for path in output.iterdir():
            if path.is_file() and path not in (target, temp) and not path.name.startswith("live.sqlite"):
                archive.write(path, path.name)
    temp.replace(target)


def run(args, plan):
    from suffix_lab.long60 import load_bundle
    config = json.loads((ROOT / "configs/long64.json").read_text(encoding="utf8"))
    cases = plan["cases"]
    if args.smoke:
        cohorts = dict.fromkeys(case["cohort"] for case in cases)
        cases = [next(case for case in cases if case["cohort"] == cohort) for cohort in cohorts]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    store = Store(output / "live.sqlite", {"plan": PLAN_SHA256, "revision": args.revision,
                                          "config": config, "ids": [case["id"] for case in cases]})
    atomic_json(output / "plan.json", {**plan, "cases": cases})
    status = dict(status="failed", planned_cases=len(cases), revision=args.revision, plan_sha256=PLAN_SHA256)
    try:
        print(f"Plan locked: {len(cases)} cases, {sum(len(actions(case)) for case in cases)} model proposals", flush=True)
        if any(store.get("case", case["id"]) is None and store.get("failure", case["id"]) is None for case in cases):
            bundle = load_bundle(config, output)
            runtime = json.loads((output / "runtime.json").read_text(encoding="utf8"))
            if runtime["models"] != plan["models"] or set(bundle.eos) != set(plan["eos"]):
                raise ValueError("Pretrained models or EOS IDs differ from the frozen source")
            atomic_json(output / "visibility.json", visibility_probe(cases, bundle.drafter))
            for index, case in enumerate(cases, 1):
                if store.get("case", case["id"]) or store.get("failure", case["id"]):
                    continue
                completed = []
                arms = list(actions(case).items())
                random.Random("soft-anchor-order-v1:" + case["id"]).shuffle(arms)
                print(f"[{index}/{len(cases)}] {case['id']} {case['cohort']}", flush=True)
                try:
                    for arm, mask in arms:
                        completed.append(execute_arm(case, arm, mask, bundle))
                        print(f"  {arm}: accepted={completed[-1]['metrics']['accepted']}", flush=True)
                except ReferenceMismatch as error:
                    store.put("failure", case["id"], dict(id=case["id"], cohort=case["cohort"], arm=arm,
                                                         evidence=error.evidence, partial_arms=completed))
                    print("  Quarantined paired case due to numerical/reference disagreement", flush=True)
                else:
                    store.put("case", case["id"], dict(id=case["id"], prompt_id=case["prompt_id"], dataset=case["dataset"],
                                                      cohort=case["cohort"], arms=completed))
                if index % 5 == 0:
                    checkpoint(output, store, {**status, "status": "running", "completed_cases": len(store.rows("case"))})
        status["status"] = "complete" if store.rows("case") else "no_audited_cases"
        if status["status"] != "complete":
            raise RuntimeError("Every case failed the canonical reference audit")
    except BaseException as error:
        status["error"] = str(error)
        raise
    finally:
        status.update(completed_cases=len(store.rows("case")), quarantined_cases=len(store.rows("failure")))
        checkpoint(output, store, status)
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--check-input", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = read_plan(args.input)
    if args.check_input:
        print(json.dumps(dict(cases=len(plan["cases"]), cohorts=plan["sample_counts"],
                              proposals=sum(len(actions(case)) for case in plan["cases"]))))
        return
    if not args.output or not args.revision:
        parser.error("--output and --revision are required for the model run")
    run(args, plan)


if __name__ == "__main__":
    main()
