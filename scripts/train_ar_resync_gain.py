"""Fit a portable COPY utility head on the completed AR-AR64 trace."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from benchmark_causal_resync import (
    FEATURES, collect_cases, create_examples, make_events, split_prompts, training_idf,
)


def policy_stats(events, actions, scores, delta, threshold):
    chosen = []
    for event in events:
        indices = np.arange(event["begin"], event["end"])
        indices = indices[scores[indices] >= threshold]
        if len(indices):
            indices = indices[actions[indices, 0] == actions[indices[0], 0]]
            chosen.append(indices[np.argmax(scores[indices])])
    values = delta[chosen]
    return dict(triggers=len(chosen), delta_floor=int(values.sum()),
                wins=int((values > 0).sum()), losses=int((values < 0).sum()))


def predict(trees, intercept, features):
    score = intercept
    for tree in trees:
        index = 0
        while len(tree[index]) > 1:
            feature, threshold, left, right = tree[index]
            index = left if features[feature] <= threshold else right
        score += tree[index][0]
    return score


def train(archive, output):
    cases, config = collect_cases(archive)
    split = split_prompts(cases)
    events, attrition = make_events(cases, config)
    idf, unknown = training_idf(events, split)
    features, actions, labels = create_examples(events, idf, unknown)
    groups = {part: [e for e in events if split[e["prompt"]] == part]
              for part in ("train", "validation", "test")}
    train_ids = np.concatenate([np.arange(e["begin"], e["end"]) for e in groups["train"]])
    base = np.repeat([e["fresh_accept"] for e in events],
                     [e["end"] - e["begin"] for e in events])
    delta = labels[:, 5] - base
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                         min_samples_leaf=80, l2_regularization=3,
                                         early_stopping=False, random_state=17)
    model.fit(features[train_ids], delta[train_ids])
    scores = model.predict(features)
    grid = {threshold: policy_stats(groups["validation"], actions, scores, delta, threshold)
            for threshold in (0, .05, .1, .2, .4, .6, .8, 1, 1.2)}
    # Retain an exploratory arm with exposure on validation, even when abstaining wins.
    threshold = max((q for q, s in grid.items() if s["triggers"] >= 20),
                    key=lambda q: (grid[q]["delta_floor"], grid[q]["triggers"]))
    trees = [
        [[float(n["value"])] if n["is_leaf"] else
         [int(n["feature_idx"]), float(n["num_threshold"]), int(n["left"]), int(n["right"])]
         for n in stage[0].nodes]
        for stage in model._predictors
    ]
    intercept = float(model._baseline_prediction[0, 0])
    for index in np.linspace(0, len(features) - 1, 4096, dtype=int):
        if abs(predict(trees, intercept, features[index]) - scores[index]) > 1e-10:
            raise AssertionError("Portable predictor differs from sklearn")
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_bytes = json.dumps([cases, config], sort_keys=True, separators=(",", ":")).encode("utf-8")
    data = dict(features=FEATURES, intercept=intercept, trees=trees, idf=idf,
                unknown=unknown, threshold=threshold, validation_grid=grid,
                test=policy_stats(groups["test"], actions, scores, delta, threshold),
                trace_sha256=sha256(trace_bytes).hexdigest(), sklearn=sklearn.__version__,
                excluded_prompt_ids=sorted(split), attrition=attrition,
                split_sizes={k: len(v) for k, v in groups.items()})
    payload = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(gzip.compress(payload, mtime=0) if output.suffix == ".gz" else payload)
    print("Frozen threshold", threshold, "validation", grid[threshold], "test", data["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train(args.archive, args.output)
