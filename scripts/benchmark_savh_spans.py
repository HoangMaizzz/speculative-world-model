"""Compare sliding-window append and dense endpoint scoring on saved SAVH features."""

import argparse
import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F


PIN = "5bcaf16e3a202a9eb05b0995559bde575d4c9fe5fd3ccab02314a110cfb83fc5"
SEEDS = (42, 123, 2026)
ARMS = {
    "hgb_scalar": (15, "hgb"),
    "mlp_scalar": (15, "mlp"),
    "mlp_hidden": (48, "mlp"),
    "conv_hidden": (48, "conv"),
    "biaffine_hidden": (48, "biaffine"),
    "attention_hidden": (48, "attention"),
}
TARGETS = (.70, .75, .80, .85, .90)
BANDS = ("4", "5-6", "7+")
PROFILES = {
    "uniform": (0., 0., 0.),
    "strict_long": (0., .03, .06),
    "stricter_long": (0., .05, .10),
    "prefer_long": (0., -.03, -.06),
}
SEED_LENGTH = 4
BUDGET = 16


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def longest(source, reference):
    previous = [0] * (len(reference) + 1)
    answer = [0] * len(source)
    for i in reversed(range(len(source))):
        previous = [1 + previous[j + 1] if token == source[i] else 0
                    for j, token in enumerate(reference)] + [0]
        answer[i] = max((n for j, n in enumerate(previous[:-1]) if j != i), default=0)
    return np.asarray(answer)


@lru_cache(maxsize=None)
def dense_rows(size, cap):
    return np.asarray([(i, i + length) for length in range(SEED_LENGTH, cap + 1)
                       for i in range(size - length + 1)], dtype=np.int64).reshape(-1, 2)


def ranking(rows, scores):
    return np.lexsort((rows[:, 1], rows[:, 0], -scores))


def choose_seeds(rows, scores, count):
    keep, deferred = [], []
    for idx in ranking(rows, scores):
        start = int(rows[idx, 0])
        if all(abs(start - int(rows[j, 0])) >= 3 for j in keep):
            keep.append(idx)
            if len(keep) == count:
                return np.asarray(keep)
        else:
            deferred.append(idx)
    return np.asarray((keep + deferred)[:count], dtype=np.int64)


def append_rows(seeds, size, cap):
    rows = set()
    for start, end in seeds.tolist():
        for left in range(max(0, end - cap), start + 1):
            for right in range(end, min(size, left + cap) + 1):
                if (left, right) != (start, end):
                    rows.add((left, right))
    return np.asarray(sorted(rows), dtype=np.int64).reshape(-1, 2)


def pack(rows, scores):
    used = 0
    selected = []
    for index in ranking(rows, scores):
        start, end = map(int, rows[index])
        length = end - start
        mask = ((1 << length) - 1) << start
        if used & mask or used.bit_count() + length > BUDGET:
            continue
        used |= mask
        selected.append(index)
        if used.bit_count() == BUDGET:
            break
    return np.asarray(selected, dtype=np.int64)


def match_labels(oracle, rows):
    return oracle[rows[:, 0]] >= np.diff(rows, axis=1)[:, 0]


def covered(rows):
    mask = 0
    for start, end in rows.tolist():
        mask |= ((1 << (end - start)) - 1) << start
    return mask.bit_count()


@dataclass
class Event:
    id: str
    prompt: str
    dataset: str
    features: np.ndarray
    oracle: np.ndarray

    @property
    def size(self):
        return len(self.features)


def load(path, smoke):
    with ZipFile(path) as archive:
        meta = json.loads(archive.read("events.json"))
        with np.load(io.BytesIO(archive.read("features.npz")), allow_pickle=False) as data:
            features = np.concatenate([data["scalar"], data["hidden"]], axis=1)
        if features.shape[1] != 48 or not np.isfinite(features).all():
            raise ValueError("Unexpected SAVH feature layout")
        groups = {}
        oracles = {}
        indices = {}
        for fold in ("train", "validation", "test"):
            eligible = [i for i, e in enumerate(meta) if e["fold"] == fold
                        and min(len(e["src"]), len(e["ref"])) >= 8]
            if smoke:
                rng = np.random.default_rng(42)
                selected = rng.choice(eligible, min(192 if fold == "train" else 64, len(eligible)), replace=False)
                eligible = sorted(selected.tolist())
            indices[fold] = eligible
            for i in eligible:
                oracles[i] = longest(meta[i]["src"], meta[i]["ref"])
        for length in (4, 8):
            with np.load(io.BytesIO(archive.read(f"input_any_{length}.npz")), allow_pickle=False) as saved:
                for i, start, label in zip(saved["groups"], saved["starts"], saved["labels"], strict=True):
                    if i in oracles and (oracles[i][start] >= length) != bool(label):
                        raise ValueError("Saved and recomputed labels differ")
        for fold, eligible in indices.items():
            events = []
            for i in eligible:
                e = meta[i]
                offset, size = e["offset"], len(e["src"])
                x = features[offset:offset + size]
                if len(x) != size:
                    raise ValueError("Truncated feature slice")
                events.append(Event(e["id"], e["prompt"], e["dataset"], x, oracles[i]))
            groups[fold] = events
    prompt_sets = [set(e.prompt for e in events) for events in groups.values()]
    if any(a & b for i, a in enumerate(prompt_sets) for b in prompt_sets[i + 1:]):
        raise ValueError("Prompt leakage across splits")
    return groups


class SpanHead(nn.Module):
    def __init__(self, channels, kind, cap, width=32):
        super().__init__()
        self.kind = kind
        self.project = nn.Sequential(nn.Linear(channels, width), nn.LayerNorm(width), nn.GELU())
        self.length = nn.Embedding(cap - 3, 8)
        if kind == "conv":
            self.context = nn.Conv1d(width, width, 3, padding=1)
        if kind == "attention":
            self.context = nn.MultiheadAttention(width, 4, batch_first=True)
        if kind == "biaffine":
            self.left = nn.Linear(width, 16)
            self.right = nn.Linear(width, 16)
            self.pool_score = nn.Linear(width + 8, 1)
        else:
            self.score = nn.Sequential(nn.Linear(3 * width + 8, 32), nn.GELU(), nn.Linear(32, 1))

    def encode(self, features, sizes):
        z = self.project(features)
        invalid = torch.arange(z.shape[1], device=z.device)[None] >= sizes[:, None]
        z = z.masked_fill(invalid[..., None], 0)
        if self.kind == "conv":
            z = z + F.gelu(self.context(z.transpose(1, 2)).transpose(1, 2))
        if self.kind == "attention":
            z = z + self.context(z, z, z, key_padding_mask=invalid, need_weights=False)[0]
        prefix = F.pad(z.cumsum(dim=1), (0, 0, 1, 0))
        if self.kind == "biaffine":
            return prefix, self.left(z), self.right(z)
        return prefix, z, z

    def score_rows(self, encoded, rows):
        prefix, left, right = encoded
        start, end = rows.unbind(dim=-1)
        length = end - start
        batch = torch.arange(prefix.shape[0], device=prefix.device)[:, None]
        pooled = (prefix[batch, end] - prefix[batch, start]) / length.clamp_min(1)[..., None]
        left = left[batch, start]
        right = right[batch, (end - 1).clamp_min(0)]
        width = self.length((length - 4).clamp_min(0))
        if self.kind == "biaffine":
            return (left * right).sum(dim=-1) / 4 + self.pool_score(torch.cat([pooled, width], -1)).squeeze(-1)
        return self.score(torch.cat([left, right, pooled, width], -1)).squeeze(-1)


def hgb_features(x, rows):
    start, end = rows.T
    width = end - start
    idx = np.arange(width.max())
    valid = idx < width[:, None]
    values = x[np.minimum(start[:, None] + idx, (end - 1)[:, None])]
    mean = (values * valid[..., None]).sum(axis=1) / width[:, None]
    return np.column_stack((mean, values.min(axis=1), values.max(axis=1), x[start], x[end - 1], width))


def train_arrays(events, channels, cap, device):
    size = max(e.size for e in events)
    n = len(dense_rows(size, cap))
    x = np.zeros((len(events), size, channels), dtype=np.float32)
    rows = np.zeros((len(events), n, 2), dtype=np.int64)
    labels = np.zeros((len(events), n), dtype=np.float32)
    sizes, counts = [], []
    for i, e in enumerate(events):
        r = dense_rows(e.size, cap)
        x[i, :e.size] = e.features[:, :channels]
        rows[i, :len(r)] = r
        labels[i, :len(r)] = match_labels(e.oracle, r)
        sizes.append(e.size)
        counts.append(len(r))
    tensors = tuple(torch.as_tensor(a, device=device) for a in (x, rows, labels, sizes))
    return (*tensors, np.asarray(counts))


def macro_ap(scores, rows, labels):
    lengths = rows[..., 1] - rows[..., 0]
    return float(np.mean([average_precision_score(labels[lengths == L], scores[lengths == L])
                          for L in np.unique(lengths) if L >= 4 and labels[lengths == L].any()]))


@torch.inference_mode()
def validation_ap(head, x, rows, labels, sizes, batch):
    head.eval()
    scores = [head.score_rows(head.encode(x[i:i + batch], sizes[i:i + batch]), rows[i:i + batch]).cpu().numpy()
              for i in range(0, len(x), batch)]
    return macro_ap(np.concatenate(scores), rows.cpu().numpy(), labels.cpu().numpy())


def fit(train, validation, channels, kind, seed, cap, device, smoke):
    rng = np.random.default_rng(seed)
    if kind == "hgb":
        xs, ys = [], []
        for event in train:
            rows = dense_rows(event.size, cap)
            rows = rows[rng.choice(len(rows), size=min(len(rows), 48), replace=False)]
            xs.append(hgb_features(event.features[:, :channels], rows))
            ys.append(match_labels(event.oracle, rows))
        head = HistGradientBoostingClassifier(max_iter=8 if smoke else 65, max_leaf_nodes=15, random_state=seed)
        head.fit(np.concatenate(xs), np.concatenate(ys))
        return head, None

    torch.manual_seed(seed)
    x, rows, labels, sizes, counts = train_arrays(train, channels, cap, device)
    vx, vr, vy, vs, _ = train_arrays(validation, channels, cap, device)
    head = SpanHead(channels, kind, cap).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.01)
    best_ap, best_weights = -1, None
    batch_size = 32 if smoke else 96
    for _ in range(4 if smoke else 8):
        head.train()
        for batch in np.array_split(rng.permutation(len(train)), (len(train) + batch_size - 1) // batch_size):
            sampled = (rng.random((len(batch), 64)) * counts[batch, None]).astype(np.int64)
            ids = torch.as_tensor(batch, device=device)
            sampled = torch.as_tensor(sampled, device=device)
            logits = head.score_rows(head.encode(x[ids], sizes[ids]), rows[ids[:, None], sampled])
            loss = F.binary_cross_entropy_with_logits(logits, labels[ids[:, None], sampled])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        ap = validation_ap(head, vx, vr, vy, vs, 96)
        if ap > best_ap:
            best_ap = ap
            best_weights = {key: value.detach().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_weights)
    return head.eval(), best_ap


class Scorer:
    def __init__(self, model, channels, kind, device, cap):
        self.model, self.channels, self.kind = model, channels, kind
        self.device, self.cap = device, cap

    def encode(self, event):
        x = np.ascontiguousarray(event.features[:, :self.channels])
        if self.kind == "hgb":
            return x
        x = torch.as_tensor(x, device=self.device)[None]
        size = torch.tensor([event.size], device=self.device)
        return self.model.encode(x, size)

    def raw(self, encoded, rows):
        if not len(rows):
            return np.empty(0)
        if self.kind == "hgb":
            return self.model.decision_function(hgb_features(encoded, rows))
        indices = torch.as_tensor(rows, device=self.device)[None]
        return self.model.score_rows(encoded, indices)[0].cpu().numpy()


def probabilities(logits, rows, calibration):
    slope, offsets = calibration
    linear = logits * slope + offsets[rows[:, 1] - rows[:, 0] - 4]
    return 1 / (1 + np.exp(-np.clip(linear, -60, 60)))


def calibrate(proposals, cap):
    eye = np.eye(cap - 3)
    x = np.concatenate([np.column_stack((p.logits, eye[p.rows[:, 1] - p.rows[:, 0] - 4])) for p in proposals])
    y = np.concatenate([p.labels for p in proposals]).astype(float)

    def objective(weights):
        logits = x @ weights
        loss = (np.logaddexp(0, logits) - y * logits).mean() + .0001 * np.dot(weights, weights)
        gradient = x.T @ (1 / (1 + np.exp(-np.clip(logits, -60, 60))) - y) / len(y) + .0002 * weights
        return loss, gradient

    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] + [(None, None)] * (x.shape[1] - 1))
    if not fit.success:
        raise ValueError(f"Calibration failed: {fit.message}")
    return float(fit.x[0]), fit.x[1:]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def pipeline(scorer, event, route, cap, seeds, instrument=False, policy=None):
    parts = {key: 0.0 for key in ("encode", "candidates", "score", "select")}

    def step(key, fn, *args):
        if not instrument:
            return fn(*args)
        begin = time.perf_counter_ns()
        result = fn(*args)
        synchronize(scorer.device)
        parts[key] += (time.perf_counter_ns() - begin) / 1e6
        return result

    encoded = step("encode", scorer.encode, event)
    scored = 0
    if route == "append":
        seed_rows = step("candidates", dense_rows, event.size, SEED_LENGTH)
        seed_logits = step("score", scorer.raw, encoded, seed_rows)
        keep = step("candidates", choose_seeds, seed_rows, seed_logits, seeds)
        extension = step("candidates", append_rows, seed_rows[keep], event.size, cap)
        extension_logits = step("score", scorer.raw, encoded, extension)
        rows = np.concatenate((seed_rows[keep], extension))
        logits = np.concatenate((seed_logits[keep], extension_logits))
        scored = len(seed_rows) + len(extension)
    else:
        rows = step("candidates", dense_rows, event.size, cap if route == "dense" else SEED_LENGTH)
        logits = step("score", scorer.raw, encoded, rows)
        scored = len(rows)
    selected = None
    if policy is not None:
        calibration, profile, threshold = policy
        p = step("score", probabilities, logits, rows, calibration)
        selected = step("select", retain, rows, p, profile, threshold)
    return rows, logits, selected, scored, parts


@dataclass
class Proposal:
    rows: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    scored: int


def gather(scorer, events, route, cap, seeds):
    proposals = []
    for event in events:
        rows, logits, _, scored, _ = pipeline(scorer, event, route, cap, seeds)
        proposals.append(Proposal(rows, logits, match_labels(event.oracle, rows), scored))
    return proposals


def bands(lengths):
    return np.where(lengths == 4, 0, np.where(lengths <= 6, 1, 2))


def adjusted_scores(rows, p, profile):
    return p - np.asarray(PROFILES[profile])[bands(rows[:, 1] - rows[:, 0])]


def retain(rows, p, profile, threshold):
    scores = adjusted_scores(rows, p, profile)
    eligible = np.flatnonzero(scores >= threshold)
    return eligible[pack(rows[eligible], scores[eligible])]


def ranked_picks(proposals, events, p, profile):
    chosen = []
    for index, (proposal, event, prob) in enumerate(zip(proposals, events, p, strict=True)):
        scores = adjusted_scores(proposal.rows, prob, profile)
        lengths = proposal.rows[:, 1] - proposal.rows[:, 0]
        chosen.extend((float(scores[i]), bool(proposal.labels[i]), int(lengths[i]), index, event.prompt)
                      for i in pack(proposal.rows, scores))
    return sorted(chosen, key=lambda row: -row[0])


def candidate_metrics(proposals, events, p):
    oracle_tokens = sum(covered(dense_rows(e.size, 4)[e.oracle[:e.size - 3] >= 4]) for e in events)
    possible = sum(covered(a.rows[a.labels]) for a in proposals)
    return {
        "oracle_tokens": oracle_tokens,
        "oracle_candidate_coverage": possible / max(1, oracle_tokens),
        "mean_candidates_scored": float(np.mean([a.scored for a in proposals])),
        "mean_candidates_retained": float(np.mean([len(a.rows) for a in proposals])),
        "candidate_macro_ap_conditional": macro_ap(np.concatenate(p), np.concatenate([a.rows for a in proposals]),
                                                   np.concatenate([a.labels for a in proposals])),
    }


def curve(picks, n_events, oracle_tokens):
    spans_ok = tokens_ok = tokens = 0
    events, prompts = set(), set()
    for i, (p, correct, length, event, prompt) in enumerate(picks, 1):
        spans_ok += correct
        tokens_ok += length * correct
        tokens += length
        events.add(event)
        prompts.add(prompt)
        if i == len(picks) or p > picks[i][0]:
            yield dict(threshold=p, spans=i, exact_spans=spans_ok, exact_tokens=tokens_ok,
                       selected_tokens=tokens, span_precision=spans_ok / i, token_precision=tokens_ok / tokens,
                       token_recall=tokens_ok / max(1, oracle_tokens), fired_events=len(events),
                       event_coverage=len(events) / n_events, fired_prompts=len(prompts))


def select_threshold(points, goal, smoke):
    feasible = [p for p in points if p["token_precision"] >= goal
                and p["span_precision"] >= goal
                and p["spans"] >= (6 if smoke else 40) and p["fired_prompts"] >= (3 if smoke else 6)]
    return max(feasible, key=lambda p: (p["exact_tokens"], p["token_precision"]), default=None)


def retention_details(picks, cut, n_events):
    selected_counts = np.zeros(n_events, dtype=int)
    by_length = {length: [0, 0] for length in sorted({row[2] for row in picks})}
    for p, correct, length, event, _ in picks:
        if p < cut:
            break
        selected_counts[event] += 1
        by_length[length][0] += 1
        by_length[length][1] += correct
    return {"suffix_counts_by_number_of_spans": np.bincount(selected_counts, minlength=5).tolist(),
            "selected_by_length": {str(L): {"spans": count, "exact_spans": exact,
                                           "precision": exact / count if count else None}
                                   for L, (count, exact) in by_length.items()}}


def prompt_totals(picks, cut, events):
    counts = {e.prompt: [0, 0, 0] for e in events}
    for e in events:
        counts[e.prompt][2] += 1
    for p, correct, length, _, prompt in picks:
        if p >= cut:
            counts[prompt][0] += length * correct
            counts[prompt][1] += length
    return counts


def paired_interval(dense, append):
    prompts = sorted(dense)
    rng = np.random.default_rng(42)
    bootstrap = np.concatenate([rng.choice(group, size=(2000, len(group)), replace=True)
                                for domain in sorted({p.split(":")[0] for p in prompts})
                                if (group := [i for i, p in enumerate(prompts) if p.split(":")[0] == domain])], axis=1)
    a, b = (np.asarray([m[p] for p in prompts]) for m in (append, dense))
    delta = a - b
    draws = delta[bootstrap, 0].sum(axis=1) / a[bootstrap, 2].sum(axis=1)
    return {"append_minus_dense_exact_tokens_per_event": float(delta[:, 0].sum() / a[:, 2].sum()),
            "prompt_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist()}


def route_report(scorer, route, calibration_events, gate, test, args):
    raw_cal, raw_val, raw_test = [gather(scorer, events, route, args.max_length, args.seeds)
                                  for events in (calibration_events, gate, test)]
    calibration = calibrate(raw_cal, args.max_length)
    val_p, test_p = [[probabilities(a.logits, a.rows, calibration) for a in raw]
                     for raw in (raw_val, raw_test)]
    val_meta = candidate_metrics(raw_val, gate, val_p)
    test_meta = candidate_metrics(raw_test, test, test_p)
    profiles = ("uniform",) if route == "fixed4" else PROFILES
    v_picks = {profile: ranked_picks(raw_val, gate, val_p, profile) for profile in profiles}
    t_picks = {profile: ranked_picks(raw_test, test, test_p, profile) for profile in profiles}
    v_points = {profile: list(curve(picks, len(gate), val_meta["oracle_tokens"]))
                for profile, picks in v_picks.items()}
    t_points = {profile: list(curve(picks, len(test), test_meta["oracle_tokens"]))
                for profile, picks in t_picks.items()}
    points, clusters = {}, {}
    timing_policy = (calibration, "uniform", -1.)
    timing_source = "unfiltered_fallback"
    for family, options in (("common", ("uniform",)), ("length_aware", profiles)):
        points[family], clusters[family] = {}, {}
        for goal in TARGETS:
            val_options = [(profile, p) for profile in options
                           if (p := select_threshold(v_points[profile], goal, args.smoke)) is not None]
            best = max(val_options, key=lambda item: (item[1]["exact_tokens"], item[0] == "uniform",
                                                       item[1]["token_precision"]), default=None)
            value = None
            if best is not None:
                profile, v = best
                cut = v["threshold"]
                t = next((p for p in reversed(t_points[profile]) if p["threshold"] >= cut), None)
                value = {"profile": profile, "probability_floors_by_band": dict(zip(
                    BANDS, (cut + d for d in PROFILES[profile]), strict=True)),
                    "validation": v, "test": t,
                    "test_retention": retention_details(t_picks[profile], cut, len(test))}
                clusters[family][str(goal)] = prompt_totals(t_picks[profile], cut, test)
                if family == "length_aware" and goal == TARGETS[0]:
                    timing_policy = (calibration, profile, cut)
                    timing_source = "length_aware_validation_70"
            points[family][str(goal)] = value
    uniform = t_points["uniform"]
    sample = np.unique(np.linspace(0, len(uniform) - 1, min(len(uniform), 101), dtype=int))
    return {
        **test_meta, "calibration": {"slope": calibration[0], "intercepts_by_length": calibration[1].tolist()},
        "unfiltered_common": uniform[-1], "operating_points": points,
        "uniform_test_curve_descriptive": [uniform[i] for i in sample],
        "uniform_best_test_at_precision_descriptive": {
            str(goal): select_threshold(uniform, goal, args.smoke) for goal in TARGETS},
        "timing_policy_source": timing_source,
    }, clusters, timing_policy


def latency(scorer, events, routes, cap, seeds, repeats, policies):
    for event in events[:4]:
        for route in routes:
            pipeline(scorer, event, route, cap, seeds, policy=policies[route])
    stages = {route: [] for route in routes}
    elapsed = {route: [] for route in routes}
    for rep in range(repeats):
        for event in events:
            for route in (routes if rep % 2 == 0 else routes[::-1]):
                synchronize(scorer.device)
                begin = time.perf_counter_ns()
                pipeline(scorer, event, route, cap, seeds, policy=policies[route])
                synchronize(scorer.device)
                elapsed[route].append((time.perf_counter_ns() - begin) / 1e6)
    for event in events:
        for route in routes:
            stages[route].append(pipeline(scorer, event, route, cap, seeds, instrument=True,
                                          policy=policies[route])[-1])
    return {route: {"p50_ms": float(np.median(elapsed[route])),
                    "p95_ms": float(np.quantile(elapsed[route], .95)),
                    "events": len(events), "repeats": repeats,
                    "stages_mean_ms_separate_pass": {key: float(np.mean([x[key] for x in stages[route]]))
                                                     for key in stages[route][0]}}
            for route in routes}


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def run(args):
    started = time.perf_counter()
    identity = digest(args.input)
    if identity != PIN:
        raise ValueError(f"Expected SAVH full ZIP SHA-256 {PIN}, got {identity}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested. Enable the Kaggle GPU accelerator or use --device cpu.")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    groups = load(args.input, args.smoke)
    train, validation, test = (groups[key] for key in ("train", "validation", "test"))
    train_tokens = np.concatenate([e.features for e in train])
    center = train_tokens.mean(axis=0)
    scale = np.maximum(train_tokens.std(axis=0), 1e-6)
    del train_tokens
    for events in groups.values():
        for event in events:
            event.features = (event.features - center) / scale
    prompts = sorted({e.prompt for e in validation})
    calibration_prompts = {p for domain in sorted({e.dataset for e in validation})
                           for p in [p for p in prompts if any(e.prompt == p and e.dataset == domain
                                                              for e in validation)][::2]}
    calibration = [e for e in validation if e.prompt in calibration_prompts]
    gate = [e for e in validation if e.prompt not in calibration_prompts]
    if not calibration or not gate:
        raise ValueError("Need separate prompt groups for calibration and threshold selection")
    rng = np.random.default_rng(42)
    timing_events = [test[i] for i in sorted(rng.choice(len(test), min(len(test), args.latency_events), replace=False))]
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint_sha256": identity, "mode": "smoke" if args.smoke else "full", "device": args.device,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch": torch.__version__, "cap": args.max_length, "seeds_per_suffix": args.seeds,
              "threshold_profiles": {name: dict(zip(BANDS, offsets, strict=True))
                                     for name, offsets in PROFILES.items()},
              "splits": {key: {"events": len(events), "prompts": len({e.prompt for e in events})}
                         for key, events in groups.items()}, "calibration_prompts": len(calibration_prompts),
              "runs": {}, "complete": False}
    atomic_json(args.output / "summary.json", result)
    print("Input checked; starting heads", result["splits"], flush=True)
    rows_csv = []
    for name, (channels, kind) in ARMS.items():
        if args.only and name not in args.only:
            continue
        seeds = (42,) if args.smoke or kind == "hgb" else SEEDS
        for seed in seeds:
            run_start = time.perf_counter()
            run_device = torch.device("cpu") if kind == "hgb" else device
            model, val_ap = fit(train, validation, channels, kind, seed, args.max_length, run_device, args.smoke)
            fit_s = time.perf_counter() - run_start
            scorer = Scorer(model, channels, kind, run_device, args.max_length)
            routes = ("fixed4", "append", "dense")
            scores, clusters, policies = {}, {}, {}
            for route in routes:
                scores[route], clusters[route], policies[route] = route_report(scorer, route, calibration, gate, test, args)
            timing = latency(scorer, timing_events, routes, args.max_length, args.seeds, args.repeats, policies)
            unfiltered = {route: (policy[0], "uniform", -1.) for route, policy in policies.items()}
            common_timing = latency(scorer, timing_events, routes, args.max_length, args.seeds, args.repeats,
                                    unfiltered)
            for route, report in scores.items():
                report["latency"] = timing[route]
                report["latency_unfiltered"] = common_timing[route]
                for family, operating in report["operating_points"].items():
                    for goal, point in operating.items():
                        row = {"arm": name, "seed": seed, "route": route, "policy_family": family, "goal": goal,
                               "device": str(run_device), "fit_s": fit_s, "p50_ms": timing[route]["p50_ms"],
                               "p95_ms": timing[route]["p95_ms"], "timing_policy_source": report["timing_policy_source"],
                               "p50_unfiltered_ms": common_timing[route]["p50_ms"],
                               "p95_unfiltered_ms": common_timing[route]["p95_ms"],
                               "oracle_candidate_coverage": report["oracle_candidate_coverage"],
                               "mean_candidates_scored": report["mean_candidates_scored"],
                               "status": ("no_validation_gate" if point is None else
                                          "no_test_selections" if point["test"] is None else "evaluated")}
                        if point is not None:
                            row.update(profile=point["profile"], locked_threshold=point["validation"]["threshold"],
                                       validation_span_precision=point["validation"]["span_precision"],
                                       validation_token_precision=point["validation"]["token_precision"],
                                       **{f"floor_{b}": p for b, p in point["probability_floors_by_band"].items()},
                                       **(point["test"] or {}))
                        rows_csv.append(row)
            paired = {family: {goal: paired_interval(clusters["dense"][family][goal], clusters["append"][family][goal])
                               for goal in clusters["dense"][family].keys() & clusters["append"][family].keys()}
                      for family in ("common", "length_aware")}
            key = f"{name}_{seed}"
            result["runs"][key] = dict(device=str(run_device), fit_s=fit_s, wall_s=time.perf_counter() - run_start,
                                       validation_ap=val_ap, routes=scores, paired=paired)
            atomic_json(args.output / "summary.json", result)
            fields = list(dict.fromkeys(field for row in rows_csv for field in row))
            with (args.output / "comparison.csv").open("w", encoding="utf8", newline="") as file:
                writer = csv.DictWriter(file, fields)
                writer.writeheader()
                writer.writerows(rows_csv)
            print(f"{key} {result['runs'][key]['wall_s']:.1f}s; "
                  + ", ".join(f"{r} p50={timing[r]['p50_ms']:.2f}ms" for r in routes), flush=True)
    result.update(complete=True, wall_s=time.perf_counter() - started)
    atomic_json(args.output / "summary.json", result)
    print(f"Complete in {result['wall_s']:.1f}s -> {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-length", type=int, choices=(8, 12, 16), default=8)
    parser.add_argument("--seeds", type=int, choices=(4, 8, 12, 16), default=12)
    parser.add_argument("--latency-events", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", nargs="+", choices=ARMS)
    args = parser.parse_args()
    if args.latency_events < 1 or args.repeats < 1:
        parser.error("Positive --latency-events and --repeats required")
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
