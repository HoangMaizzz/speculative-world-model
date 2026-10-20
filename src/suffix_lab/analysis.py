"""Prompt-cluster uncertainty, paired comparisons and ten diagnostic figures."""
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
from .store import Store, atomic_json
from .metrics import event_gains


def clustered_summary(values, prompts, samples=2000, seed=42):
    """Bootstrap prompt clusters, preserving all states/events within each cluster."""
    import numpy as np
    grouped = defaultdict(list)
    for value, prompt in zip(values, prompts):
        if value is not None and np.isfinite(value):
            grouped[prompt].append(float(value))
    if not grouped:
        return {"n": 0, "prompts": 0, "mean": None, "median": None, "ci_low": None, "ci_high": None}
    groups = [grouped[k] for k in sorted(grouped)]
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        indices = rng.integers(0, len(groups), len(groups))
        boot.append(sums[indices].sum()/counts[indices].sum())
    flat = [x for g in groups for x in g]
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"n": len(flat), "prompts": len(groups), "mean": statistics.mean(flat),
            "median": statistics.median(flat), "ci_low": float(low), "ci_high": float(high)}


def analyze(output, bootstrap_samples=2000):
    # Matplotlib reads MPLBACKEND during its first import, before use() can run.
    # Kaggle's inline backend belongs to the notebook, not this PNG exporter.
    os.environ["MPLBACKEND"] = "Agg"
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    output = Path(output)
    destination = output / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    store = Store(output / "run.sqlite")
    one, two, refs = store.rows("test1"), store.rows("test2"), store.rows("reference")
    status = {k: bool(store.get("meta", k+"_complete")) for k in ("references", "test1", "test2")}
    token_rows, rows1, rows2 = [], [], []
    for row in one:
        m = row["metrics"]
        rows1.append({k: row[k] for k in ("id", "state_id", "prompt_id", "dataset", "source", "physical_phase", "offset")} |
                     {k: v for k, v in m.items() if not isinstance(v, list)} |
                     {"draft_latency_ms": row["draft"]["stats"]["latency_ms"],
                      "draft_forwards": row["draft"]["stats"]["total_forwards"],
                      "verify_compute_ms": row["feedback"].get("latency_ms"),
                      "verify_wall_ms": row["feedback"].get("wall_latency_ms"),
                      "progress_bin": row.get("progress_bin", "unknown") })
        for i in range(m["first_rejection"], len(row["draft"]["tokens"])):
            token_rows.append({"id": row["id"], "prompt_id": row["prompt_id"], "dataset": row["dataset"],
                "source": row["source"], "distance": i-m["first_rejection"]+1,
                "correct": int(m["reference_matches"][i]), "old_agrees": int(m["old_matches"][i]),
                "verifier_confidence": row["feedback"]["probability"][i],
                "drafter_confidence": row["draft"]["confidence"][i], "unmask_order": row["draft"]["order"][i]})
    for row in two:
        row.setdefault("redrafter", "dllm_top1")  # Read legacy checkpoints without mixing baselines.
        row.setdefault("arm_family", "oracle" if row["oracle"] else "nonoracle")
    fresh = {(r["event_id"], r["redrafter"]): r for r in two if r["method"] == "fresh_L"}
    def compute_work(row):
        verify = row.get("verification", {}).get("latency_ms")
        return row["stats"]["latency_ms"]+verify if verify is not None else None
    def difference(a, b):
        return a-b if a is not None and b is not None else None
    for row in two:
        pair = (row["event_id"], row["redrafter"])
        if pair not in fresh:
            continue
        baseline = fresh[pair]
        gain = row["metrics"]["accepted_prefix"] - baseline["metrics"]["accepted_prefix"]
        rows2.append({k: row[k] for k in ("id", "event_id", "prompt_id", "dataset", "source", "method", "oracle",
                        "first_rejection", "horizon", "physical_phase", "keep_ratio", "redrafter", "arm_family")} |
                     row["metrics"] | {"gain": gain,
                        "progress_bin": row.get("progress_bin", "unknown"),
                        "probe_gain": row.get("probe_gain"),
                        "probe_accepted_prefix": row.get("probe_accepted_prefix"),
                        "t_draft_wall_ms": row["stats"]["latency_ms"],
                        "t_verify_compute_ms": row.get("verification", {}).get("latency_ms"),
                        "t_verify_wall_ms": row.get("verification", {}).get("wall_latency_ms"),
                        "t_draft_plus_verify_compute_ms": compute_work(row),
                        "draft_plus_verify_compute_delta_ms": difference(compute_work(row), compute_work(baseline)),
                        "t_selector_ms_event": row.get("selector_ms_event"),
                        "t_wall_ms": row.get("pipeline_wall_ms"),
                        "pipeline_wall_delta_ms": difference(row.get("pipeline_wall_ms"), baseline.get("pipeline_wall_ms")), "win": int(gain > 0), "tie": int(gain == 0), "loss": int(gain < 0),
                        "forwards": row["stats"]["total_forwards"], "unmask_forwards": row["stats"]["unmask_steps"],
                        "latency_ms": row["stats"]["latency_ms"],
                        "mutable_latency_ms": row["stats"].get("mutable_latency_ms", 0.),
                        "forward_delta": row["stats"]["total_forwards"]-baseline["stats"]["total_forwards"],
                        "latency_delta_ms": row["stats"]["latency_ms"]-baseline["stats"]["latency_ms"]})
    frame1, frame2, tokens = pd.DataFrame(rows1), pd.DataFrame(rows2), pd.DataFrame(token_rows)
    for name, frame in (("test1", frame1), ("test2", frame2), ("suffix_tokens", tokens)):
        frame.to_csv(destination / (name+".csv"), index=False)
    summaries = []

    def summarize(frame, keys, metrics, study):
        if frame.empty:
            return
        for group, part in frame.groupby(keys, dropna=False):
            group = group if isinstance(group, tuple) else (group,)
            for metric in metrics:
                summary = clustered_summary(part[metric].tolist(), part.prompt_id.tolist(), bootstrap_samples)
                summaries.append({"study": study, **dict(zip(keys, group)), "metric": metric, **summary})

    summarize(frame1, ["dataset", "source"], ["accepted_prefix", "suffix_match_rate", "suffix_match_count",
        "direct_rejoin_length", "island_count", "max_island_length", "all_accepted",
        "suffix_at_least_1", "suffix_at_least_2", "suffix_at_least_4", "suffix_at_least_8"], "test1")
    summarize(frame2, ["dataset", "source", "redrafter", "method"], ["accepted_prefix", "full_block_accuracy", "retained_precision",
        "remasked_accuracy", "fully_correct", "gain", "win", "tie", "loss", "forwards", "latency_ms",
        "forward_delta", "latency_delta_ms"], "test2")
    summarize(frame2, ["dataset", "source", "redrafter", "first_rejection", "horizon", "method"],
        ["accepted_prefix", "gain", "forwards", "latency_ms"], "test2_rejection_strata")
    summarize(frame1, ["physical_phase", "source"], ["suffix_match_rate", "accepted_prefix"], "phase_audit")
    summarize(frame1, ["dataset", "source", "progress_bin"], ["accepted_prefix", "suffix_match_rate"], "progress_audit")
    paired = []
    by_state = defaultdict(dict)
    for row in rows1:
        by_state[row["state_id"]][row["source"]] = row
    for state, pair in by_state.items():
        if len(pair) != 2:
            continue
        d, a = pair["dllm_top1"], pair["ar_greedy"]
        paired.append({"state_id": state, "prompt_id": d["prompt_id"], "dataset": d["dataset"],
            "suffix_match_count_delta": d["suffix_match_count"]-a["suffix_match_count"],
            "accepted_prefix_delta": d["accepted_prefix"]-a["accepted_prefix"],
            "suffix_match_rate_delta": (d["suffix_match_rate"]-a["suffix_match_rate"])
                if d["suffix_match_rate"] is not None and a["suffix_match_rate"] is not None else None})
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(destination / "paired.csv", index=False)
    summarize(paired_frame, ["dataset"], ["suffix_match_count_delta", "suffix_match_rate_delta", "accepted_prefix_delta"], "paired_dllm_minus_ar")
    summarize(tokens, ["dataset", "source", "distance"], ["correct", "old_agrees"], "distance")
    if not tokens.empty:
        dllm_tokens = tokens[tokens.source == "dllm_top1"]
        summarize(dllm_tokens, ["dataset", "unmask_order"], ["correct"], "unmask_order")
    pd.DataFrame(summaries).to_csv(destination / "summary.csv", index=False)
    feature_scores = []
    if not tokens.empty:
        for (dataset, source), part in tokens.groupby(["dataset", "source"]):
            for feature in ("verifier_confidence", "drafter_confidence", "old_agrees"):
                labels, scores = part.correct, part[feature]
                feature_scores.append({"dataset": dataset, "source": source, "feature": feature, "n": len(part),
                    "prevalence": float(labels.mean()), "auroc": float(roc_auc_score(labels, scores)) if labels.nunique() == 2 else None,
                    "auprc": float(average_precision_score(labels, scores)) if labels.any() else None,
                    "brier": float(brier_score_loss(labels, scores)), "uncertainty": "point estimate; correlated tokens"})
    atomic_json(destination / "feature_scores.json", feature_scores)

    def figure(filename, title, draw, available):
        height = max(5, .22*frame2.groupby(["source", "redrafter", "method"]).ngroups) if filename.startswith("07_") and not frame2.empty else 5
        fig, ax = plt.subplots(figsize=(10, height))
        if available:
            draw(ax)
        else:
            ax.text(.5, .5, "No eligible records yet (partial run or empty suffix)", ha="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(destination / filename, dpi=150)
        plt.close(fig)

    def line_groups(frame, x, y, ax):
        for source, part in frame.groupby("source"):
            means = part.groupby(x, observed=True)[y].mean()
            ax.plot(means.index.astype(float), means.values, marker=".", label=source)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()

    figure("01_rejection.png", "First rejection (33 = all accepted)",
        lambda ax: [ax.hist(p.first_rejection, bins=np.arange(.5, 34.5), alpha=.5, label=s) for s, p in frame1.groupby("source")] and ax.legend(), not frame1.empty)
    figure("02_suffix_by_rejection.png", "Suffix match rate, conditional on nonempty rejected suffix",
        lambda ax: line_groups(frame1, "first_rejection", "suffix_match_rate", ax), not frame1.empty)
    islands = [(row["source"], length) for row in one for length in row["metrics"]["island_lengths"]]
    figure("03_islands.png", "Matched island lengths after first rejection",
        lambda ax: [ax.hist([v for s, v in islands if s == source], bins=np.arange(.5, 33.5), alpha=.5, label=source)
                    for source in ("dllm_top1", "ar_greedy")] and ax.legend(), bool(islands))
    figure("04_ar_vs_dllm.png", "True suffix agreement vs distance after first rejection",
        lambda ax: line_groups(tokens, "distance", "correct", ax), not tokens.empty)
    d_tokens = tokens[tokens.source == "dllm_top1"] if not tokens.empty else tokens
    figure("05_unmask_order.png", "dLLM unmask order vs true suffix correctness (0-based)",
        lambda ax: line_groups(d_tokens, "unmask_order", "correct", ax), not d_tokens.empty)
    def confidence_plot(ax):
        temp = tokens.copy()
        temp["confidence_bin"] = (temp.verifier_confidence*10).clip(0, 9).astype(int)/10+.05
        line_groups(temp, "confidence_bin", "correct", ax)
    figure("06_verifier_confidence.png", "Old-context draft-token probability vs true suffix correctness",
        confidence_plot, not tokens.empty)
    def bars(ax):
        means = frame2.groupby(["source", "redrafter", "method"]).gain.mean().sort_values()
        means.index = [" / ".join(x) for x in means.index]
        ax.barh(means.index, means.values)
        ax.set_xlabel("Mean gain vs own event/redrafter fresh-L (inspect stratified CSV)")
        ax.tick_params(axis="y", labelsize=7)
    figure("07_methods.png", "Repair methods; oracle shown as diagnostic only", bars, not frame2.empty)
    def scatter(ax, x):
        for method, part in frame2.groupby(["source", "redrafter", "method"]):
            ax.scatter(part[x].mean(), part.accepted_prefix.mean(), label="/".join(method), s=30)
        ax.set_xlabel(x + " (cold replay; no online speedup)")
        ax.set_ylabel("Accepted prefix tokens")
        ax.legend(fontsize=5, ncol=2)
    figure("08_forwards.png", "Accepted length vs all drafter forwards", lambda ax: scatter(ax, "forwards"), not frame2.empty)
    figure("09_latency.png", "Accepted length vs measured drafter latency", lambda ax: scatter(ax, "latency_ms"), not frame2.empty)
    def ratios(ax):
        temp = frame2[~frame2.oracle & frame2.method.str.contains("@")].copy()
        temp["family"] = temp.method.str.split("@").str[0]
        for family, part in temp.groupby(["source", "redrafter", "family"]):
            means = part.groupby("keep_ratio").gain.mean()
            ax.plot(means.index, means.values, marker=".", label="/".join(family))
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xlabel("Actual KEEP fraction (rounded count)")
        ax.set_ylabel("Accepted-prefix gain vs same-event fresh-L")
        ax.legend(fontsize=7)
    figure("10_keep_gain.png", "KEEP fraction vs accepted-prefix gain", ratios, not frame2.empty)
    event_causes = []
    for (event, redrafter), base in fresh.items():
        arms = [x for x in two if x["event_id"] == event and x["redrafter"] == redrafter]
        event_causes.append({"event_id": event, "redrafter": redrafter, "source": base["source"], "dataset": base["dataset"], **event_gains(arms)})
    atomic_json(destination / "event_causes.json", event_causes)
    cross, controls, quality, repairability = [], [], [], []
    by_arm = defaultdict(dict)
    for row in rows2:
        by_arm[(row["event_id"], row["method"])][row["redrafter"]] = row
    raw = {(r["event_id"], r["redrafter"], r["method"]): r for r in two}
    for (event, method), pair in by_arm.items():
        if set(pair) != {"dllm_top1", "ar_greedy"}:
            continue
        d, a = pair["dllm_top1"], pair["ar_greedy"]
        dp, ap = raw[event, "dllm_top1", method], raw[event, "ar_greedy", method]
        if dp.get("anchors") != ap.get("anchors"):
            raise AssertionError("Cross-redrafter anchor plans differ")
        cross.append({"event_id": event, "prompt_id": d["prompt_id"], "dataset": d["dataset"],
            "source": d["source"], "method": method,
            "dllm_gain": d["gain"], "ar_gain": a["gain"], "gain_interaction": d["gain"]-a["gain"],
            "dllm_draft_delta_ms": d["latency_delta_ms"], "ar_draft_delta_ms": a["latency_delta_ms"],
            "draft_delta_interaction_ms": d["latency_delta_ms"]-a["latency_delta_ms"],
            "dllm_pipeline_wall_delta_ms": d["pipeline_wall_delta_ms"],
            "ar_pipeline_wall_delta_ms": a["pipeline_wall_delta_ms"]})
    for row in two:
        if row["arm_family"] != "conditioning":
            continue
        old = raw.get((row["event_id"], row["redrafter"], "future_old"))
        controls.append({k: row.get(k) for k in ("event_id", "prompt_id", "dataset", "source", "redrafter",
            "method", "uses_reference", "probe_horizon", "probe_accepted_prefix", "probe_gain",
            "content_changed_from_old", "future_anchors_visible_to_probe")} |
            {"anchor_count": len(row["keep"]), "positions": row["keep"],
             "content_probe_delta_vs_old": row["probe_accepted_prefix"]-old["probe_accepted_prefix"]})
    if not frame1.empty:
        for (dataset, source), part in frame1.groupby(["dataset", "source"]):
            quality.append({"dataset": dataset, "source": source, "proposals": len(part),
                "accepted_prefix_mean": part.accepted_prefix.mean(), "full_accept_rate": part.all_accepted.mean(),
                "draft_wall_ms_mean": part.draft_latency_ms.mean(), "draft_forwards_mean": part.draft_forwards.mean()})
            eligible = part[part.suffix_length > 0]
            repairability.append({"dataset": dataset, "source": source, "all_proposals": len(part),
                "nonempty_suffix_events": len(eligible),
                "conditional_macro_match_rate": eligible.suffix_match_rate.mean() if len(eligible) else None,
                "conditional_pooled_match_rate": eligible.suffix_match_count.sum()/eligible.suffix_length.sum() if len(eligible) else None,
                "conditional_reusable_event_rate": (eligible.suffix_match_count > 0).mean() if len(eligible) else None,
                "correct_suffix_tokens_per_proposal": part.suffix_match_count.sum()/len(part)})
    for name, records in (("cross_redrafter", cross), ("conditioning", controls),
                          ("draft_quality", quality), ("repairability", repairability)):
        pd.DataFrame(records).to_csv(destination/(name+".csv"), index=False)
    summarize(pd.DataFrame(cross), ["dataset", "source", "method"],
              ["gain_interaction", "draft_delta_interaction_ms"], "cross_redrafter_interaction")
    summarize(pd.DataFrame(controls), ["dataset", "source", "redrafter", "method"],
              ["probe_gain", "content_probe_delta_vs_old"], "future_content_control")
    pd.DataFrame(summaries).to_csv(destination/"summary.csv", index=False)
    from .experiment import progress_bin
    prefix_coverage = []
    for ref in refs:
        counts = {b: sum(progress_bin(o, len(ref["tokens"])) == b for o in ref["offsets"])
                  for b in ("early", "middle", "late")}
        prefix_coverage.append({"prompt_id": ref["id"], "dataset": ref["dataset"],
            "reference_length": len(ref["tokens"]), **counts,
            "missing_progress_bins": [b for b, n in counts.items() if not n]})
    atomic_json(destination/"prefix_coverage.json", prefix_coverage)
    report = {"status": status, "reference_prompts": len(refs), "eligible_states": sum(len(r["offsets"]) for r in refs),
        "no_eligible_state_prompts": sum(not r["offsets"] for r in refs), "test1_proposals": len(one),
        "paired_states": len(paired), "nonempty_rejection_events": sum(r["metrics"]["suffix_length"] > 0 for r in one),
        "sampled_test2_events": len(store.get("meta", "test2_sample") or []),
        "complete_test2_events": len(store.rows("event_complete")), "test2_arms": len(two),
        "reusable_suffix_events": sum(r["metrics"]["suffix_match_count"] > 0 for r in one),
        "complete_redrafter_pairs": len(store.rows("pair_complete")),
        "oracle_selector_gap_pairs": sum(x["oracle_selector_gap"] for x in event_causes),
        "no_bounded_correct_keep_gain_pairs": sum(x["no_bounded_correct_keep_gain"] for x in event_causes),
        "no_observed_gain_pairs": sum(x["no_observed_gain"] for x in event_causes),
        "conditioning_arms": len(controls), "cross_redrafter_comparisons": len(cross)}
    atomic_json(destination / "report.json", report)
    text = "# Suffix diagnostic report\n\n" + "\n".join(f"- {k}: {v}" for k, v in report.items())
    text += "\n\nMean, median, N and 95% prompt-cluster bootstrap intervals are in summary.csv. "
    text += "Paired dLLM-minus-AR comparisons use common states; conditional suffix-rate differences require nonempty suffixes in both. "
    text += "cross_redrafter.csv reports each arm gain and time delta against its OWN redrafter fresh-L before comparing architectures. conditioning.csv uses identical future positions/count, with a strict AR causal negative control. sampling_coverage.json records available/selected/empty source/benchmark/rejection/progress strata. Read benchmark/source/redrafter/rejection/horizon strata before comparing aggregates. Confidence curves and feature scores are descriptive, without fitted selectors.\n\n"
    text += "Oracle A is not a global upper bound. B is a bounded search run identically for both redrafters, regardless of A gain. Correct/wrong future-content controls use offline labels and are not deployable selectors or automatically recovered old tokens. "
    text += "A selector gap suggests studying feedback/selection; failure of bounded oracles suggests checking geometry audits, mask patterns and infilling ability, without identifying a unique cause.\n\n"
    text += "Timing: verifier stats use CUDA compute-stage events excluding Accelerate dispatch copies; wall_latency_ms is preserved separately. Stage time excludes inter-GPU copies and orchestration outside selected stages and is not end-to-end latency. Drafter plots use one-GPU synchronized cold-replay wall time with no peer transfer. Do not interpret wall-minus-stage as measured communication time.\n\n"
    text += "All latency is synchronized independent cold replay, including completed-prefix reconstruction; mutable-only time is also exported. "
    text += "Identical anchor positions AND contents share one measured inference within an event/redrafter; aliases are dependent observations. Selector time is event-level, oracle acquisition/search is excluded from per-arm latency. "
    text += "Every unique repaired candidate is reverified, with a hard target/reference consistency check. This is an offline cold replay, without an online rollback benchmark and no evidence of end-to-end speedup. Drafts are fixed horizon even after draft EOS; target reference stops at EOS and states need a full horizon plus bonus token. "
    text += "Partial runs are not completed pilot results. Decide whether to build an online method only after complete results and uncertainty show useful paired suffix headroom, oracle gain and measurable compute savings.\n"
    text += "Differences compare these trained checkpoints and schedules; they do not isolate architecture from training or other model differences.\n"
    (destination / "REPORT.md").write_text(text, encoding="utf-8")
    store.close()
    return report
