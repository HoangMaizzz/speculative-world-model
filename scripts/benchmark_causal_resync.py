"""Train a COPY decision on prefix-only replay of native AR-AR draft trajectories."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LENGTHS = (4, 8, 16)
PROBES = (4, 8, 16)
MATCHES = (1, 2, 3, 4, 6, 8)
THRESHOLDS = (0.0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95)
SEED = 17
FEATURES = (
    "t", "source_index", "shift", "abs_shift", "old_length", "tail_room", "copy_length",
    "match", "ambiguity", "token_matches", "match_unique", "match_idf", "token_idf",
    "seen_in_probe", "copy_unique_fraction", "copy_idf", "first_copy_repeats",
    "prior_accepted", "offset", "corrected_idf", "corrected_token_in_match_context",
)


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def before_eos(tokens, eos):
    return tokens[:next((i for i, token in enumerate(tokens) if token in eos), len(tokens))]


def split_prompts(cases):
    split = {}
    for dataset in ("math500", "gsm8k"):
        ids = sorted(c["id"] for c in cases if c["dataset"] == dataset)
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError(f"Expected 100 unique {dataset} prompts in the complete replay archive")
        random.Random(SEED).shuffle(ids)
        for name, group in (("train", ids[:60]), ("validation", ids[60:80]), ("test", ids[80:])):
            split.update(dict.fromkeys(group, name))
    return split


def collect_cases(archive):
    if Path(archive).is_dir():
        summary = json.loads((Path(archive) / "summary.json").read_bytes())
        cases = [json.loads(line) for line in (Path(archive) / "cases.jsonl").read_bytes().splitlines()]
    else:
        with ZipFile(archive) as zip_file:
            summary = json.load(zip_file.open("summary.json"))
            cases = [json.loads(line) for line in zip_file.open("cases.jsonl")]
    cfg = summary["manifest"]
    if (summary["status"], len(cases), cfg["draft_length"], cfg["temperature"]) != ("complete", 200, 64, 0):
        raise ValueError("Expected the complete 200-prompt greedy AR-AR64 archive")
    return cases, cfg


def make_events(cases, cfg):
    events, attrition = [], Counter()
    eos = set(cfg["eos_ids"])
    for case in cases:
        gold = case["ar"]["token_ids"]
        spec = case["spec"]["token_ids"]
        shared = lcp(spec, gold)
        rounds = case["spec"]["rounds"]
        for index, row in enumerate(rounds):
            decision = row["decision"]
            rejection = decision["first_rejection"]
            if rejection is None:
                continue
            attrition["native_rejections"] += 1
            offset = row["offset"] + len(decision["committed_tokens"])
            if index + 1 == len(rounds) or offset >= len(gold):
                attrition["no_next_round_or_reference"] += 1
                continue
            if offset > shared:
                attrition["outside_greedy_parity"] += 1
                continue
            following = rounds[index + 1]
            if following["offset"] != offset or spec[offset - 1] != decision["correction"]:
                raise ValueError(f"Corrected prefix is not the next draft input: {case['id']}/{index}")
            budget = min(cfg["draft_length"], cfg["max_new_tokens"] - offset)
            fresh = following["proposal"]
            if len(fresh) > budget or (len(fresh) < budget and fresh[-1] not in eos):
                raise ValueError("Unexpected early stop of the fresh draft")
            if len(gold) < offset + budget and gold[-1] not in eos:
                raise ValueError("Reference ends inside the candidate window without EOS")
            events.append(dict(prompt=case["id"], dataset=case["dataset"], round=index, offset=offset,
                               old=before_eos(row["proposal"][rejection + 1:], eos), fresh=fresh,
                               observed=before_eos(fresh, eos), gold=gold[offset:], budget=budget,
                               corrected=decision["correction"], prior_accepted=decision["accepted_draft"]))
    return events, dict(attrition)


def training_idf(events, split):
    # Count observations already visible to the drafter, only from training prompts.
    fingerprints = {}
    for event in events:
        if split[event["prompt"]] == "train":
            fingerprints.setdefault(event["prompt"], set()).update(event["old"] + event["observed"])
    freq = Counter(token for seen in fingerprints.values() for token in seen)
    total = len(fingerprints) + 1
    return {tok: float(np.log(total / (1 + count))) for tok, count in freq.items()}, float(np.log(total))


def candidates(event, idf, unknown, *, last_only=False):
    """Yield causally observable action features, before consulting reference labels."""
    old = event["old"]
    ends = {}
    for i, token in enumerate(old):
        ends.setdefault(token, []).append(i)
    corrected = event["corrected"]
    for t, token in enumerate(event["observed"][:max(PROBES)], 1):
        if last_only and t != len(event["observed"]):
            continue
        probe = event["observed"][:t]
        indices = [i for i in ends.get(token, []) if len(old) - i - 1 >= min(LENGTHS)]
        matches = {i: lcp(probe[-8:][::-1], old[max(0, i - 7):i + 1][::-1]) for i in indices}
        for i in indices:
            k = matches[i]
            match = old[i - k + 1:i + 1]
            uniqueness = sum(other >= k for other in matches.values())
            shift = t - 1 - i
            for length in LENGTHS:
                if t + length > event["budget"] or i + 1 + length > len(old):
                    continue
                tail = old[i + 1:i + 1 + length]
                feat = (t, i, shift, abs(shift), len(old), len(old) - i - 1, length,
                        k, uniqueness, len(ends[token]), len(set(match)),
                        sum(idf.get(x, unknown) for x in match) / k, idf.get(token, unknown),
                        probe.count(token), len(set(tail)) / length,
                        sum(idf.get(x, unknown) for x in tail) / length, int(tail[0] == token),
                        event["prior_accepted"], event["offset"], idf.get(corrected, unknown),
                        int(i >= k and old[i - k] == corrected))
                yield (t, i, length, k, uniqueness, shift), feat


def create_examples(events, idf, unknown):
    features, actions, labels = [], [], []
    for event in events:
        event["fresh_accept"] = lcp(event["fresh"], event["gold"])
        begin = len(actions)
        for action, feat in candidates(event, idf, unknown):
            t, i, length, *_ = action
            tail = event["old"][i + 1:i + 1 + length]
            tail_lcp = lcp(tail, event["gold"][t:])
            copy_accept = tail_lcp if t <= event["fresh_accept"] else 0
            accepted = min(t, event["fresh_accept"]) + copy_accept
            same_as_fresh = tail == event["fresh"][t:t + length]
            # Replicating a greedy fresh slice preserves the remaining trajectory in exact arithmetic.
            # A divergent COPY has known acceptance up to its end. Future tokens can only increase it.
            resumed_floor = event["fresh_accept"] if same_as_fresh else accepted
            labels.append((int(copy_accept == length), int(tail_lcp == length), copy_accept, accepted,
                           int(same_as_fresh), resumed_floor))
            actions.append(action)
            features.append(feat)
        event["begin"], event["end"] = begin, len(actions)
    return np.asarray(features, dtype=np.float32), np.asarray(actions, dtype=np.int32), np.asarray(labels, dtype=np.int32)


def stage_choices(groups, actions, probabilities, wait, parameter, length=0):
    """Select at most one COPY action per event, at the first eligible probe step."""
    selected = []
    for event in groups:
        idx = np.arange(event["begin"], event["end"])
        rows = actions[idx]
        if probabilities is None:
            allowed = (rows[:, 0] <= wait) & (rows[:, 2] == length) & (rows[:, 3] >= parameter)
        else:
            allowed = (rows[:, 0] <= wait) & (probabilities[idx] >= parameter)
        idx = idx[allowed]
        if len(idx) == 0:
            selected.append(-1)
            continue
        rows = actions[idx]
        earliest = rows[0, 0]  # Enumerated in increasing t.
        idx = idx[rows[:, 0] == earliest]
        r = actions[idx]
        key = (-r[:, 3] if probabilities is None else -probabilities[idx] * r[:, 2])
        best = np.lexsort((r[:, 1], np.abs(r[:, 5]), r[:, 4], key))[0]
        selected.append(int(idx[best]))
    return selected


def ratio(n, d):
    return float(n / d) if d else None


def evaluate(groups, selected, actions, labels, comparison):
    counts = Counter()
    per_prompt = {}
    lens = Counter()
    for event, idx in zip(groups, selected):
        prompt = per_prompt.setdefault(event["prompt"], [0, 0, 0])
        prompt[0] += 1
        counts["events"] += 1
        fresh = event["fresh_accept"]
        counts["fresh_accepted"] += fresh
        counts["fresh_draft_steps"] += len(event["fresh"])
        if idx < 0:
            continue
        t, i, length, k, ambiguity, shift = map(int, actions[idx])
        full, tail_exact, copied, accepted, same_as_fresh, resumed = map(int, labels[idx])
        delta = accepted - fresh
        matched_fresh = min(fresh, t + length)
        short = min(fresh, t)
        saved = len(event["fresh"]) - t
        equal_budget_saved = min(saved, length)
        matched_delta, resumed_delta = accepted - matched_fresh, resumed - fresh
        compared_delta = {"stop_full": delta, "same_length": matched_delta, "resume_floor": resumed_delta}[comparison]
        counts.update(triggers=1, copied_submitted=length, copied_accepted=copied, full_copy=full,
                      tail_exact=tail_exact, bridge_wrong=t > fresh, zero_copy_accepted=copied == 0,
                      shifted=shift != 0, shifted_full=shift != 0 and full,
                      win=compared_delta > 0, tie=compared_delta == 0, loss=compared_delta < 0,
                      accept_delta=compared_delta, stopped_vs_full_delta=delta,
                      same_length_accept_delta=matched_delta, resumed_accept_delta_floor=resumed_delta,
                      exactly_same_as_fresh=same_as_fresh,
                      divergent_fully_accepted=not same_as_fresh and full,
                      stop_after_probe_accept_delta=accepted - short, serial_steps_skipped=saved,
                      equal_length_steps_skipped=equal_budget_saved, probe_steps=t)
        prompt[1] += compared_delta
        prompt[2] += equal_budget_saved
        lens[length] += 1
    out = dict(counts)
    for key in ("triggers", "copied_submitted", "copied_accepted", "full_copy", "accept_delta",
                "equal_length_steps_skipped", "serial_steps_skipped"):
        out.setdefault(key, 0)
    out.update(copy_exact_rate=ratio(out["full_copy"], out["triggers"]),
               copy_token_accept_rate=ratio(out["copied_accepted"], out["copied_submitted"]),
               trigger_rate=ratio(out["triggers"], out["events"]),
               accepted_delta_per_event=ratio(out["accept_delta"], out["events"]),
               equal_length_steps_per_event=ratio(out["equal_length_steps_skipped"], out["events"]),
               selected_lengths=dict(sorted(lens.items())))
    return out, per_prompt


def rank_metrics(groups, actions, labels, probabilities):
    ids = np.concatenate([np.arange(e["begin"], e["end"]) for e in groups])
    y, scores = labels[ids, 0], probabilities[ids]
    result = {"overall": dict(candidates=len(ids), positives=int(y.sum()),
                              ap=float(average_precision_score(y, scores)),
                              auc=float(roc_auc_score(y, scores)), brier=float(brier_score_loss(y, scores)))}
    for length in LENGTHS:
        mask = actions[ids, 2] == length
        result[str(length)] = dict(candidates=int(mask.sum()), positives=int(y[mask].sum()),
                                   ap=float(average_precision_score(y[mask], scores[mask])))
    compared, trained_top1, rule_top1 = 0, 0, 0
    for e in groups:
        ids = np.arange(e["begin"], e["end"])
        for t in range(1, max(PROBES) + 1):
            step = ids[(actions[ids, 0] == t) & (actions[ids, 2] == 4)]
            if len(step) < 2 or not labels[step, 0].any():
                continue
            compared += 1
            trained_top1 += int(labels[step[np.argmax(probabilities[step])], 0])
            options = actions[step]
            best = np.lexsort((options[:, 1], np.abs(options[:, 5]), options[:, 4], -options[:, 3]))[0]
            rule_top1 += int(labels[step[best], 0])
    result["ambiguous_positive_steps_L4"] = dict(steps=compared, model_correct=trained_top1, rule_correct=rule_top1)
    return result


def opportunity(groups, actions, labels):
    stats = Counter(events=len(groups))
    for e in groups:
        ids = np.arange(e["begin"], e["end"])
        stats["candidate_events"] += len(ids) > 0
        if len(ids) == 0:
            continue
        positive = ids[labels[ids, 0] == 1]
        stats["tail_only_positive_events"] += bool(labels[ids, 1].any())
        stats["positive_events"] += len(positive) > 0
        stats["candidate_rows"] += len(ids)
        stats["positive_rows"] += len(positive)
        for wait in PROBES:
            for k in MATCHES:
                present = (actions[positive, 0] <= wait) & (actions[positive, 3] >= k)
                stats[f"t{wait}_k{k}"] += bool(present.any())
        stats["same_length_accept_gain_events"] += bool(
            (labels[ids, 3] > np.minimum(e["fresh_accept"], actions[ids, 0] + actions[ids, 2])).any())
        stats["full_fresh_accept_gain_events"] += bool((labels[ids, 3] > e["fresh_accept"]).any())
    return dict(stats)


def tune(family, groups, actions, labels, probabilities):
    options = ((wait, k, length) for wait in PROBES for k in MATCHES for length in LENGTHS)
    if probabilities is not None:
        options = ((wait, threshold, 0) for wait in PROBES for threshold in THRESHOLDS)
    scored = []
    for wait, argument, length in options:
        selected = stage_choices(groups, actions, probabilities, wait, argument, length)
        stats, _ = evaluate(groups, selected, actions, labels, "stop_full")
        scored.append(dict(family=family, wait=wait, argument=argument, length=length, stats=stats))
    return scored


def pick(scored, tolerated_loss, metric):
    n = scored[0]["stats"]["events"]
    feasible = [p for p in scored if p["stats"].get(metric, 0) >= -tolerated_loss * n]
    return max(feasible, key=lambda p: (p["stats"]["equal_length_steps_skipped"],
                                        p["stats"].get(metric, 0), p["stats"]["full_copy"])) if feasible else None


def bootstrap(per_prompt):
    data = np.array(list(per_prompt.values()), dtype=float)
    rng = np.random.default_rng(SEED)
    samples = data[rng.integers(len(data), size=(2000, len(data)))].sum(axis=1)
    intervals = np.percentile(samples[:, 1:] / samples[:, :1], [2.5, 97.5], axis=0)
    return dict(accepted_delta_ci=intervals[:, 0].tolist(), equal_length_steps_ci=intervals[:, 1].tolist())


def run(archive, output):
    cases, manifest = collect_cases(archive)
    split = split_prompts(cases)
    events, attrition = make_events(cases, manifest)
    idf, unknown = training_idf(events, split)
    features, actions, labels = create_examples(events, idf, unknown)
    groups = {name: [e for e in events if split[e["prompt"]] == name]
              for name in ("train", "validation", "test")}
    train_ids = np.concatenate([np.arange(e["begin"], e["end"]) for e in groups["train"]])
    models = {
        "logistic": (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=200)), list(range(len(FEATURES)))),
        "hgb": (HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, min_samples_leaf=60,
                                                l2_regularization=2, early_stopping=False, random_state=SEED),
                list(range(len(FEATURES)))),
        "hgb_geometry": (HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, min_samples_leaf=60,
                                                         l2_regularization=2, early_stopping=False, random_state=SEED),
                         [*range(7), FEATURES.index("prior_accepted"), FEATURES.index("offset")]),
    }
    print(f"Trace events {len(events)}, actions {len(actions)}, training actions {len(train_ids)}", flush=True)
    grid = {"suffix_rule": tune("suffix_rule", groups["validation"], actions, labels, None)}
    predictions, ranking = {"suffix_rule": None}, {}
    for name, (model, columns) in models.items():
        print(f"Fitting {name}", flush=True)
        inputs = features[:, columns]
        model.fit(inputs[train_ids], labels[train_ids, 0])
        p = model.predict_proba(inputs)[:, 1]
        predictions[name] = p
        ranking[name] = {which: rank_metrics(groups[which], actions, labels, p) for which in ("validation", "test")}
        grid[name] = tune(name, groups["validation"], actions, labels, p)

    policies = {}
    comparisons = {"stop_full": "accept_delta", "same_length": "same_length_accept_delta",
                   "resume_floor": "resumed_accept_delta_floor"}
    for comparison, metric in comparisons.items():
        policies[comparison] = {}
        for tolerance in (0.05, 0.25):
            reports = {}
            for name, candidates_on_val in grid.items():
                config = pick(candidates_on_val, tolerance, metric)
                if config is None:
                    continue
                chosen = stage_choices(groups["test"], actions, predictions[name], config["wait"],
                                       config["argument"], config["length"])
                stats, per_prompt = evaluate(groups["test"], chosen, actions, labels, comparison)
                datasets = {}
                for ds in ("math500", "gsm8k"):
                    paired = [(e, a) for e, a in zip(groups["test"], chosen) if e["dataset"] == ds]
                    datasets[ds] = evaluate([e for e, _ in paired], [a for _, a in paired], actions, labels, comparison)[0]
                reports[name] = dict(validation=config, test=stats, datasets=datasets, test_ci=bootstrap(per_prompt))
            policies[comparison][str(tolerance)] = reports
    identity = {key: manifest[key] for key in ("models", "precision", "temperature", "draft_length", "max_new_tokens")}
    result = dict(archive_sha256=sha256(archive.read_bytes()).hexdigest(), trace_model_config=identity,
                  split=split, attrition=attrition, horizon=dict(probe_steps=PROBES, copy_lengths=LENGTHS),
                  features=list(FEATURES), opportunities={k: opportunity(g, actions, labels) for k, g in groups.items()},
                  test_opportunities_by_dataset={ds: opportunity([e for e in groups["test"] if e["dataset"] == ds],
                                                                  actions, labels) for ds in ("math500", "gsm8k")},
                  ranking=ranking, policies=policies, validation_grid=grid)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    joblib.dump(dict(models=models, features=FEATURES, idf=idf, unknown=unknown,
                     policy={comparison: {t: {name: data["validation"] for name, data in report.items()}
                                          for t, report in by_tolerance.items()}
                             for comparison, by_tolerance in policies.items()}), output / "copy_heads.joblib", compress=3)
    print(output / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.output)
