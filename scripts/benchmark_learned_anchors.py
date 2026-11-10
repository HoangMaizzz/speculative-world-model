"""Compare frozen, conditional, stacked and duration heads for exact SAVH anchors."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

import benchmark_savh_spans as b


FLOORS = (.50, .40)
GOALS = (.55, .60, .65, .70, .75)
CUTS = np.round(np.arange(.25, .851, .025), 3)


def maximal(rows):
    end = -1
    result = []
    for i, j in sorted(set(map(tuple, rows.tolist())), key=lambda r: (r[0], -r[1])):
        if j > end:
            result.append((i, j))
            end = j
    return np.asarray(result, dtype=int).reshape(-1, 2)


def regions(seed, floor):
    result = []
    start = None
    for i, score in enumerate(seed):
        if score < floor or (start is not None and i + 4 - start > 8):
            if start is not None:
                result.append((start, i + 3, "score" if score < floor else "cap"))
                start = None
        if score >= floor and start is None:
            start = i
    if start is not None:
        result.append((start, len(seed) + 3, "end"))
    return result


def pool(rows, zones):
    keep = np.zeros(len(rows), dtype=bool)
    for i, j, _ in zones:
        keep |= (rows[:, 0] >= i) & (rows[:, 1] <= j)
    return keep


def hgb(x, y, smoke):
    return HistGradientBoostingClassifier(max_iter=8 if smoke else 65, max_leaf_nodes=15,
                                           random_state=42).fit(x, y)


def stats(event, rows):
    return b.hgb_features(event.features[:, :15], rows)


def extension_input(event, rows):
    return np.column_stack((stats(event, rows), stats(event, np.column_stack((rows[:, 1] - 4, rows[:, 1])))))


def train_extension(events, smoke):
    rng = np.random.default_rng(123)
    x, y = [], []
    for event in events:
        r = b.dense_rows(event.size, 8)
        length = r[:, 1] - r[:, 0]
        r = r[(length > 4) & (event.oracle[r[:, 0]] >= length - 1)]
        if len(r) > 40:
            r = r[rng.choice(len(r), 40, replace=False)]
        if len(r):
            x.append(extension_input(event, r))
            y.append(b.match_labels(event.oracle, r))
    return hgb(np.concatenate(x), np.concatenate(y), smoke)


def append_logits(model, event, rows):
    ext = rows[:, 1] - rows[:, 0] > 4
    raw = np.zeros(len(rows))
    raw[ext] = model.decision_function(extension_input(event, rows[ext]))
    return raw


def chain_features(rows, q):
    lookup = {tuple(r): k for k, r in enumerate(rows.tolist())}
    stack = np.zeros((len(rows), 4))
    stack[:, :2] = 1.
    for k, (i, j) in enumerate(rows.tolist()):
        if j - i > 4:
            prev = lookup[(i, j - 1)]
            stack[k] = (q[k], min(q[k], stack[prev, 1]), stack[prev, 2] + np.log(max(q[k], 1e-12)), 1.)
    return stack


def train_hgb_arms(train, smoke):
    rng = np.random.default_rng(42)
    picked = []
    for e in train:
        rows = b.dense_rows(e.size, 8)
        picked.append(rng.choice(len(rows), min(len(rows), 48), replace=False))
    y = np.concatenate([b.match_labels(e.oracle, b.dense_rows(e.size, 8)[ids])
                        for e, ids in zip(train, picked, strict=True)])
    x = np.concatenate([stats(e, b.dense_rows(e.size, 8)[ids]) for e, ids in zip(train, picked, strict=True)])
    base = hgb(x, y, smoke)
    oof = [None] * len(train)
    prompts = [e.prompt for e in train]
    folds = StratifiedGroupKFold(n_splits=2 if smoke else 5, shuffle=True, random_state=42)
    for fold, (inside, outside) in enumerate(folds.split(np.zeros(len(train)), [e.dataset for e in train], prompts), 1):
        assert {prompts[i] for i in inside}.isdisjoint(prompts[i] for i in outside)
        ext = train_extension([train[i] for i in inside], smoke)
        for k in outside:
            e = train[k]
            rows = b.dense_rows(e.size, 8)
            q = 1. / (1. + np.exp(-np.clip(append_logits(ext, e, rows), -60, 60)))
            oof[k] = chain_features(rows, q)[picked[k]]
        print(f"OOF fold {fold} complete", flush=True)
    assert all(f is not None for f in oof)
    return base, train_extension(train, smoke), hgb(np.column_stack((x, np.concatenate(oof))), y, smoke)


class Duration(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(15, 32), nn.GELU())
        self.local = nn.Conv1d(32, 32, 3, padding=1)
        self.out = nn.Linear(32, 6)

    def forward(self, x, size):
        pos = torch.arange(x.shape[1], device=x.device)
        z = self.encode(x).masked_fill((pos >= size[:, None])[..., None], 0)
        logits = self.out(z + F.gelu(self.local(z.transpose(1, 2)).transpose(1, 2)))
        invalid = torch.arange(4, 9, device=x.device)[None, None, :] > (size[:, None] - pos)[..., None]
        return logits.masked_fill(F.pad(invalid, (1, 0), value=False), -torch.inf)


def duration_arrays(events, device):
    x = np.zeros((len(events), 63, 15), dtype=np.float32)
    y = np.zeros(x.shape[:2], dtype=np.int64)
    for i, e in enumerate(events):
        x[i, :e.size] = e.features[:, :15]
        y[i, :e.size] = np.clip(e.oracle - 3, 0, 5)
    return tuple(torch.as_tensor(a, device=device) for a in (x, y, [e.size for e in events]))


def fit_duration(train, validation, device, seed, smoke):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y, n = duration_arrays(train, device)
    vx, vy, vn = duration_arrays(validation, device)
    model = Duration().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    positions = torch.arange(x.shape[1], device=device)
    best, state = float("inf"), None
    for _ in range(2 if smoke else 10):
        model.train()
        for batch in np.array_split(rng.permutation(len(train)), (len(train) + 95) // 96):
            ids = torch.as_tensor(batch, device=device)
            mask = positions < n[ids, None] - 3
            loss = F.cross_entropy(model(x[ids], n[ids])[mask], y[ids][mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            mask = positions < vn[:, None] - 3
            loss = float(F.cross_entropy(model(vx, vn)[mask], vy[mask]))
        if loss < best:
            best, state = loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def duration_score(model, event, rows, device):
    x = torch.as_tensor(np.ascontiguousarray(event.features[:, :15]), device=device)[None]
    mass = model(x, torch.tensor([event.size], device=device))[0].softmax(-1)
    tail = mass.flip(-1).cumsum(-1).flip(-1).cpu().numpy()
    return tail[rows[:, 0], rows[:, 1] - rows[:, 0] - 3]


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def calibrate(records, arm, conditional=False):
    proposals = []
    for r in records:
        rows = r["rows"]
        keep = np.ones(len(rows), dtype=bool)
        if conditional:
            length = rows[:, 1] - rows[:, 0]
            keep = (length > 4) & (r["event"].oracle[rows[:, 0]] >= length - 1)
        proposals.append(b.Proposal(rows[keep], r["raw"][arm][keep], r["labels"][keep], int(keep.sum())))
    return b.calibrate(proposals, 8)


def build_records(events, models, durations, device):
    h, ext, stack = models
    records = []
    for event in events:
        rows = b.dense_rows(event.size, 8)
        start = time.perf_counter_ns()
        x = stats(event, rows)
        raw = {"A1": h.decision_function(x)}
        joint_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        raw["q"] = append_logits(ext, event, rows)
        q = 1. / (1. + np.exp(-np.clip(raw["q"], -60, 60)))
        stack_inputs = chain_features(rows, q)
        extension_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        raw["A3"] = stack.decision_function(np.column_stack((x, stack_inputs)))
        elapsed = {"A0": joint_ms, "A1": joint_ms, "A2": joint_ms + extension_ms,
                   "A3": joint_ms + extension_ms + (time.perf_counter_ns() - start) / 1e6}
        for key, model in durations.items():
            b.synchronize(device)
            start = time.perf_counter_ns()
            raw[key] = logit(duration_score(model, event, rows, device))
            b.synchronize(device)
            elapsed[key] = joint_ms + (time.perf_counter_ns() - start) / 1e6
        starts = np.flatnonzero(event.oracle >= 4)
        oracle = maximal(np.column_stack((starts, starts + event.oracle[starts])))
        records.append(dict(event=event, rows=rows, oracle=oracle, raw=raw, score_ms=elapsed,
                            labels=b.match_labels(event.oracle, rows)))
    return records


def prepare(records, seed_cal, extension_cal, floor):
    for r in records:
        rows, event = r["rows"], r["event"]
        seed = b.probabilities(r["raw"]["A1"], rows, seed_cal)
        start = time.perf_counter_ns()
        r["regions"] = regions(seed[:event.size - 3], floor)
        r["pool"] = pool(rows, r["regions"])
        r["region_ms"] = (time.perf_counter_ns() - start) / 1e6
        r["fixed"] = (rows[:, 1] - rows[:, 0]) == 4
        q = b.probabilities(r["raw"]["q"], rows, extension_cal)
        # Seed and extension probabilities share a monotone chain for each source start.
        r["chain"] = np.zeros(len(rows))
        lookup = {tuple(row): k for k, row in enumerate(rows.tolist())}
        for k, (i, j) in enumerate(rows.tolist()):
            r["chain"][k] = seed[k] if j - i == 4 else r["chain"][lookup[(i, j - 1)]] * q[k]
        assert all(r["chain"][k] <= r["chain"][lookup[(i, j - 1)]] + 1e-12
                   for k, (i, j) in enumerate(rows.tolist()) if j - i > 4)


def difficult_cases(records):
    counts = Counter()
    for r in records:
        event, rows = r["event"], r["rows"]
        start, end = rows.T
        length = end - start
        missed_extension = (event.oracle[start] == length - 1) & (length > 4)
        incompatible = (event.oracle[start] >= 4) & (event.oracle[end - 4] >= 4) & ~r["labels"]
        counts["true_parent_false_extension"] += int((r["pool"] & missed_extension).sum())
        counts["both_edge_windows_exact_union_wrong"] += int((r["pool"] & incompatible).sum())
        for i, j, _ in r["regions"]:
            counts["regions"] += 1
            if event.oracle[i] < 4 and np.any(event.oracle[i + 1:j - 3] >= 4):
                counts["wrong_seed_with_exact_restart"] += 1
    return dict(counts)


def calibrate_on_pool(records, arm):
    proposals = []
    for r in records:
        mask = r["fixed"] if arm == "A0" else r["pool"]
        rows = r["rows"][mask]
        proposals.append(b.Proposal(calibration_rows(rows, arm), r["raw"]["A1" if arm == "A0" else arm][mask],
                                    r["labels"][mask], 0))
    return b.calibrate(proposals, 8)


def calibration_rows(rows, arm):
    # A shared monotone mapping preserves the ordering of a duration head's prefix probabilities.
    return np.column_stack((rows[:, 0], rows[:, 0] + 4)) if arm.startswith("A4_") else rows


def scored(records, arm, calibration):
    for r in records:
        mask = r["fixed"] if arm == "A0" else r["pool"]
        rows = r["rows"][mask]
        if arm == "A2":
            p = r["chain"][mask]
        else:
            p = b.probabilities(r["raw"]["A1" if arm == "A0" else arm][mask], calibration_rows(rows, arm), calibration)
        yield r, rows, p, r["labels"][mask]


def select(rows, p, threshold, cap):
    ids = np.flatnonzero(p >= threshold)
    kept = maximal(rows[ids])
    if not cap or len(kept) <= cap:
        return kept
    scores = dict(zip(map(tuple, rows[ids].tolist()), p[ids], strict=True))
    ranked = sorted(map(tuple, kept.tolist()), key=lambda r: (-scores[r], r))
    seen, chosen = set(), []
    while ranked and len(chosen) < cap:
        best = max(ranked, key=lambda r: scores[r] * sum(i not in seen for i in range(*r)))
        ranked.remove(best)
        chosen.append(best)
        seen.update(range(*best))
    return np.asarray(chosen, dtype=int).reshape(-1, 2)


def measure(data, threshold, cap=0, cases=None, timed=False):
    counts, per_prompt = Counter(), {}
    active = set()
    for r, rows, p, _ in data:
        event = r["event"]
        begin = time.perf_counter_ns()
        keep = select(rows, p, threshold, cap)
        if timed:
            r["selection_ms"] = (time.perf_counter_ns() - begin) / 1e6
        truth = b.match_labels(event.oracle, keep)
        exact = keep[truth]
        mask = np.zeros(event.size, dtype=bool)
        for i, j in exact:
            mask[i:j] = True
        target = set(map(tuple, exact.tolist()))
        covered_run = sum(int(mask[i:j].sum()) for i, j in r["oracle"])
        counts.update(kept=len(keep), exact=len(exact), tokens_unique=int(mask.sum()), tokens_by_run=covered_run,
                      oracle_exact=sum(tuple(row) in target for row in r["oracle"].tolist()),
                      oracle_touched=sum(bool(mask[i:j].any()) for i, j in r["oracle"]),
                      empty=not len(keep), single=len(keep) == 1, multiple=len(keep) >= 2)
        for i, j in exact:
            counts[f"exact_L{j-i}"] += 1
        c = per_prompt.setdefault(event.prompt, [0, 0, 0])
        c[0] += covered_run
        c[1] += len(exact)
        c[2] += len(keep)
        if len(keep):
            active.add(event.prompt)
        if cases is not None:
            scores = dict(zip(map(tuple, rows.tolist()), p, strict=True))
            cases.append({"event": event.id, "regions": r["regions"], "threshold": threshold,
                          "kept": [{"start": i, "end": j, "score": float(scores[(i, j)]), "exact": bool(y)}
                                   for (i, j), y in zip(keep.tolist(), truth, strict=True)]})
    return {**counts, "precision": counts["exact"] / max(1, counts["kept"]), "active_prompts": len(active),
            "threshold": float(threshold), "prompt_counts": per_prompt}


def rank_report(data, arm, floor):
    for length in range(4, 9):
        pairs = [(p[(rows[:, 1] - rows[:, 0]) == length], y[(rows[:, 1] - rows[:, 0]) == length])
                 for _, rows, p, y in data]
        valid = [(p, y) for p, y in pairs if y.any()]
        if not valid:
            continue
        ys, ps = np.concatenate([y for p, y in pairs]), np.concatenate([p for p, y in pairs])
        yield {"arm": arm, "floor": floor, "length": length, "ap": float(average_precision_score(ys, ps)),
               "positive_events": len(valid), "candidates": len(ys), **{f"hit_at_{k}": sum(
                   bool(y[np.argsort(-p, kind="stable")[:k]].any()) for p, y in valid) for k in (1, 4, 8)}}


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def paired(a, b_):
    keys = sorted(a)
    rng = np.random.default_rng(42)
    by_domain = [[k for k in keys if k.split(":")[0] == d] for d in sorted({k.split(":")[0] for k in keys})]
    draws = sum(np.asarray([a[k][0] - b_[k][0] for k in g])[rng.integers(0, len(g), (2000, len(g)))].sum(1)
                for g in by_domain)
    return {"tokens_delta": sum(a[k][0] - b_[k][0] for k in keys), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def run(args):
    begin = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU or pass --device cpu")
    path = b.find_input(args.input, args.search_root)
    print(f"Input: {path}", flush=True)
    groups = b.load(path, args.smoke)
    train, val, test = groups.values()
    x = np.concatenate([e.features for e in train]); center, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    for e in train + val + test:
        e.features = (e.features - center) / scale
    cal_ids = {p for domain in sorted({e.dataset for e in val})
               for p in sorted({e.prompt for e in val if e.dataset == domain})[::2]}
    models = train_hgb_arms(train, args.smoke)
    durations = {f"A4_{seed}": fit_duration(train, val, device, seed, args.smoke)
                 for seed in ((42,) if args.smoke else (42, 123, 2026))}
    rv, test_records = (build_records(es, models, durations, device) for es in (val, test))
    cal = [r for r in rv if r["event"].prompt in cal_ids]
    gate = [r for r in rv if r["event"].prompt not in cal_ids]
    seed_cal, ext_cal = calibrate(cal, "A1"), calibrate(cal, "q", conditional=True)
    oracle = {"tokens_unique": sum(b.covered(r["oracle"]) for r in test_records),
              "tokens_by_run": sum(int(np.diff(r["oracle"], axis=1).sum()) for r in test_records),
              "maximal_runs": sum(len(r["oracle"]) for r in test_records),
              "maximal_runs_le8": sum(int((np.diff(r["oracle"], axis=1) <= 8).sum()) for r in test_records)}
    if not args.smoke:
        assert oracle == dict(tokens_unique=9623, tokens_by_run=9975, maximal_runs=1131, maximal_runs_le8=794)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"complete": False, "mode": "smoke" if args.smoke else "full", "input_sha256": b.PIN,
               "device": args.device, "events": {k: len(v) for k, v in groups.items()},
               "calibration_prompts": sorted(cal_ids), "oracle": oracle, "floors": {}}
    b.atomic_json(args.output / "manifest.json", {**summary, "floor_grid": FLOORS, "cut_grid": CUTS.tolist(),
                  "precision_goals": GOALS, "prompt_split": {k: sorted({e.prompt for e in v}) for k, v in groups.items()}})
    csv_rows, ranks = [], []
    with (args.output / "cases.jsonl").open("w", encoding="utf8") as case_file:
        for floor in FLOORS:
            for records in (cal, gate, test_records):
                prepare(records, seed_cal, ext_cal, floor)
            s = {"candidate_count": sum(int(r["pool"].sum()) for r in test_records),
                 "candidate_correct_tokens": sum(b.covered(r["rows"][r["pool"] & r["labels"]]) for r in test_records),
                 "hard_cases": difficult_cases(test_records), "arms": {}}
            for arm in ("A0", "A1", "A2", "A3", *durations):
                c = None if arm == "A2" else calibrate_on_pool(cal, arm)
                g, t = (list(scored(recs, arm, c)) for recs in (gate, test_records))
                ranks.extend(rank_report(t, arm, floor))
                by_cap = {}
                for cap in (0, 4, 8):
                    val_points = [measure(g, float(cut), cap) for cut in CUTS]
                    test_points = [measure(t, float(cut), cap) for cut in CUTS]
                    locked = {}
                    for goal in GOALS:
                        possible = [p for p in val_points if p["precision"] >= goal and
                                    p["kept"] >= (8 if args.smoke else 50) and p["active_prompts"] >= (2 if args.smoke else 6)]
                        best = max(possible, key=lambda p: (p["tokens_by_run"], p["precision"]), default=None)
                        if best is None:
                            locked[str(goal)] = None
                            continue
                        point = next(p for p in test_points if p["threshold"] == best["threshold"])
                        locked[str(goal)] = {"validation": best, "test": point}
                        csv_rows.append({"floor": floor, "arm": arm, "cap": cap, "split": "locked_test", "goal": goal,
                                         **{k: v for k, v in point.items() if k != "prompt_counts"}})
                    csv_rows.extend({"floor": floor, "arm": arm, "cap": cap, "split": "test_descriptive",
                                     **{k: v for k, v in p.items() if k != "prompt_counts"}} for p in test_points)
                    by_cap[str(cap)] = locked
                point = by_cap["0"]["0.65"] or by_cap["0"]["0.55"]
                cut = .50 if point is None else point["test"]["threshold"]
                cases = []
                measure(t, cut, cases=cases, timed=True)
                for row in cases:
                    case_file.write(json.dumps({"arm": arm, "floor": floor, **row}) + "\n")
                times = [r["score_ms"][arm] + r["region_ms"] + r["selection_ms"] for r in test_records]
                s["arms"][arm] = {"locked": by_cap, "latency_cached_ms_p50_p95": np.quantile(times, [.5, .95]).tolist(),
                                  "timing_threshold": cut, "timing_scope": "precomputed dense scores plus region and selector timing",
                                  "stage_ms_p50_p95": {stage: np.quantile(values, [.5, .95]).tolist() for stage, values in (
                                      ("scoring", [r["score_ms"][arm] for r in test_records]),
                                      ("region", [r["region_ms"] for r in test_records]),
                                      ("selection", [r["selection_ms"] for r in test_records]))}}
                print(f"floor {floor} {arm} complete", flush=True)
            s["paired_A3_minus_A1"] = {}
            for goal in GOALS:
                aa, bb = (s["arms"][arm]["locked"]["0"][str(goal)] for arm in ("A3", "A1"))
                if aa and bb:
                    s["paired_A3_minus_A1"][str(goal)] = paired(aa["test"]["prompt_counts"], bb["test"]["prompt_counts"])
            summary["floors"][str(floor)] = s
            b.atomic_json(args.output / "summary.json", summary)
            csv_file(args.output / "precision_coverage.csv", csv_rows)
            csv_file(args.output / "rank_by_length.csv", ranks)
    summary.update(complete=True, wall_s=time.perf_counter() - begin)
    b.atomic_json(args.output / "summary.json", summary)
    print(f"Complete in {summary['wall_s']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args)


if __name__ == "__main__":
    main()
