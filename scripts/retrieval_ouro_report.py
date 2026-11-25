"""Paired inference-speed report. Wall time includes warm-model prompt prefill and all decode work."""
import csv
import io
import json
from collections import Counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

ARMS = ("A", "B", "C", "D", "O16", "O4")
PAIRS = (("A", "B"), ("A", "C"), ("A", "D"), ("A", "O16"), ("A", "O4"),
         ("B", "C"), ("B", "D"), ("O16", "B"), ("O16", "C"), ("O16", "D"),
         ("O4", "B"), ("O4", "C"), ("O4", "D"))

def aggregate(rows, method):
    runs = [row["runs"][method] for row in rows]
    counts = Counter()
    for run in runs:
        counts.update(run["counts"])
    tokens = sum(len(run["tokens"]) for run in runs)
    return dict(counts=dict(counts), output_tokens=tokens, tokens_per_s=tokens*1000/counts["wall_ms"],
                accepted_per_verify=counts["accepted"]/counts["rounds"],
                emitted_per_verify=tokens/counts["rounds"],
                exact_vs_A=sum(run["tokens"] == row["runs"]["A"]["tokens"] for run,row in zip(runs,rows)))

def paired(rows, base, method):
    if not rows:
        return None
    totals = np.zeros(2)
    draws = np.zeros((2000,2))
    rng = np.random.default_rng(925)
    for domain in sorted({r["dataset"] for r in rows}):
        times = np.array([[r["runs"][a]["counts"]["wall_ms"] for a in (base,method)]
                          for r in rows if r["dataset"] == domain])
        totals += times.sum(axis=0)
        draws += times[rng.integers(len(times),size=(len(draws),len(times)))].sum(axis=1)
    return dict(n=len(rows), wall_ratio=totals[0]/totals[1],
                ci95=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist())

def describe(rows):
    pairs={}
    for a,b in PAIRS:
        equal = [r for r in rows if r["runs"][a]["tokens"] == r["runs"][b]["tokens"]]
        pairs[b+"_vs_"+a]=dict(all_prompts=paired(rows,a,b), same_output=paired(equal,a,b),
                                exact=len(equal), total=len(rows))
    return dict(arms={arm:aggregate(rows,arm) for arm in ARMS}, pairs=pairs)

def write_bundle(out,manifest,rows):
    groups={"all":rows}
    for domain in ("math500","gsm8k"):
        groups[domain]=[r for r in rows if r["dataset"]==domain]
    summary={name:describe(group) for name,group in groups.items() if group}
    summary["completed"]=len(rows)
    summary["planned"]=len(manifest["prompts"])
    summary["complete"]=summary["completed"]==summary["planned"]
    buf=io.StringIO(newline="")
    writer=csv.writer(buf)
    writer.writerow(["prompt","dataset","arm","order","output_tokens","equal_to_A","first_difference",
                     "wall_s","draft_s","verify_s","regrow_s","draft_forwards","verify_rounds","accepted","accepted_phrase"])
    for row in rows:
        base=row["runs"]["A"]["tokens"]
        for k,arm in enumerate(row["order"]):
            run=row["runs"][arm]
            seq=run["tokens"]
            cnt=run["counts"]
            diff=next((i for i,(x,y) in enumerate(zip(seq,base)) if x!=y),min(len(seq),len(base)))
            writer.writerow([row["prompt"],row["dataset"],arm,k+1,len(seq),seq==base,"" if seq==base else diff,
                             cnt["wall_ms"]/1000,cnt["draft_ms"]/1000,cnt["verify_ms"]/1000,cnt.get("regrow_ms",0)/1000,
                             cnt["draft_forwards"],cnt["rounds"],cnt["accepted"],cnt.get("accept_phrase","")])
    temp=out/"report.tmp"
    with ZipFile(temp,"w",ZIP_DEFLATED,compresslevel=3) as bundle:
        bundle.writestr("manifest.json",json.dumps(manifest))
        bundle.writestr("summary.json",json.dumps(summary,indent=1))
        bundle.writestr("per_prompt.csv",buf.getvalue())
        for row in rows:
            bundle.writestr("rows/"+row["prompt"].replace(":","_")+".json",json.dumps(row,separators=(",",":")))
    temp.replace(out/"retrieval_ouro_results.zip")
    return summary
