"""Only committed history and original prompt IDs enter online feature inputs."""
from collections import defaultdict
from suffix_lab.long60_plan import decide, until_eos, validate_draft
from suffix_lab.store import digest


def reconstruct(prompts, rounds, eos, max_new_tokens=256):
    prompts = {p["id"]: p for p in prompts}
    groups = defaultdict(list)
    for r in rounds:
        groups[r["prompt_id"]].append(r)
    if set(groups) != set(prompts) or len({r["id"] for r in rounds}) != len(rounds):
        raise ValueError("Prompt coverage or duplicate round IDs")
    events, excluded = [], []
    for pid in sorted(groups):
        p, output = prompts[pid], []
        for index, row in enumerate(sorted(groups[pid], key=lambda r: r["round_index"])):
            x = p["prompt_ids"]+output
            if (row["round_index"] != index or row["offset"] != len(output)
                    or row["prefix_digest"] != digest(x) or (output and output[-1] in eos)):
                raise ValueError(f"Invalid prefix chain: {row['id']}")
            validate_draft(row["draft"], 64, x)
            proposal = until_eos(row["draft"]["tokens"], eos)
            if row["proposal"] != proposal:
                raise ValueError("Logical EOS proposal differs")
            expected = decide(proposal, row["verification"]["top1_with_bonus"], eos,
                              max_new_tokens-len(output))
            if expected != row["decision"]:
                raise ValueError("Actual commit differs")
            feedback = row["verification"]["parallel_feedback"]
            if len(feedback["top1_with_bonus"]) != len(proposal)+1:
                raise ValueError("Parallel feedback row count")
            a, c = expected["first_rejection"], expected["correction"]
            if a is not None:
                source = until_eos(proposal[a+1:], eos, False)
                if c in eos or not source:
                    excluded.append({"id": row["id"], "reason": "terminal_correction" if c in eos else "empty_source"})
                else:
                    # References/diagnostics are deliberately absent from this object.
                    events.append({"id": row["id"], "prompt_id": pid, "dataset": row["dataset"],
                                   "offset": row["offset"], "prefix": list(x), "proposal": proposal,
                                   "accepted": a, "correction": c, "source": source,
                                   "corrected_prefix": x+proposal[:a]+[c], "parallel": feedback,
                                   "correction_emitted_within_output_budget": len(expected["committed_tokens"]) == a+1,
                                   "draft": row["draft"], "guard": row["verification"]["canonical_guard"]})
            output += expected["committed_tokens"]
        if output != p["tokens"]:
            raise ValueError("Final committed output differs")
    return events, excluded
