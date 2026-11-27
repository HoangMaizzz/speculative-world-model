"""Paired benchmark at identical prompt, output cap and 16-node draft budget."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from random import Random
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from huggingface_hub import snapshot_download
from threadpoolctl import threadpool_limits

from benchmark_dllm_ar32 import decode_ar
from benchmark_retrieval_alignment import HASHES, inputs
from benchmark_retrieval_ouro import atomic, plan
from consensus_tree_core import (DRAFT, WIDTH, FORK_LIMIT, FORK_THRESHOLD, SWITCH_THRESHOLD, P_MIN,
                                 P_WEIGHT, TOP_Q, MODES, run, sync)
from retrieval_alignment_core import Index, longest_prefix
from retrieval_ouro_report import paired
from suffix_lab.assets import ROOT, TARGET, AR
from suffix_lab.runtime import load_model, load_target, tokenizers

ARCHIVE = "consensus_tree_results.zip"
ARMS = MODES + ("greedy7b",)

def reference(p, ids, eos, limit):
    sync()
    start = perf_counter()
    r = decode_ar(p,ids,eos,limit)
    sync()
    return dict(tokens=r["token_ids"], counts=dict(rounds=r["forwards"],accepted=0,
                                                  wall_ms=(perf_counter()-start)*1000))

def describe(rows):
    result = {"arms":{},"pairs":{}}
    for arm in ARMS:
        counts=Counter()
        for row in rows:
            counts.update(row["runs"][arm]["counts"])
        n=sum(len(row["runs"][arm]["tokens"]) for row in rows)
        result["arms"][arm]=dict(counts=counts,output_tokens=n,tokens_per_s=1000*n/counts["wall_ms"],
                                 emitted_per_verify=n/counts["rounds"],
                                 accepted_per_verify=counts["accepted"]/counts["rounds"] if arm!="greedy7b" else None,
                                 exact_vs_greedy7b=sum(r["runs"][arm]["tokens"]==r["runs"]["greedy7b"]["tokens"] for r in rows))
        if arm!="base":
            same=[r for r in rows if r["runs"][arm]["tokens"]==r["runs"]["base"]["tokens"]]
            exact=[r for r in same if r["runs"][arm]["tokens"]==r["runs"]["greedy7b"]["tokens"]]
            result["pairs"][arm]=dict(all_prompts=paired(rows,"base",arm),same_output=paired(same,"base",arm),
                                      exact_to_greedy7b=paired(exact,"base",arm),
                                      exact_to_base=len(same),total=len(rows))
    return result

def write_report(out,manifest,rows):
    groups={domain:[r for r in rows if r["dataset"]==domain] for domain in ("math500","gsm8k")}
    summary={k:describe(v) for k,v in {"all":rows,**groups}.items() if v}
    summary.update(completed=len(rows),planned=len(manifest["prompts"]),complete=len(rows)==len(manifest["prompts"]))
    buff=StringIO(newline="")
    csv_out=csv.writer(buff)
    keys=("wall_ms","prefill_ms","draft_ms","verify_ms","maintenance_ms","draft_forwards","q_fill_calls",
          "rounds","accepted","tree_nodes","feedback_rows","actions","hits","extra","verifier_branch_nodes")
    csv_out.writerow(["id","dataset","arm","order","tokens","first_diff_base","first_diff_greedy7b",*keys])
    for row in rows:
        runs=row["runs"]
        for i,arm in enumerate(row["order"]):
            run_=runs[arm]
            seq=run_["tokens"]
            diffs=["" if seq==runs[b]["tokens"] else longest_prefix(seq,runs[b]["tokens"]) for b in ("base","greedy7b")]
            csv_out.writerow([row["prompt"],row["dataset"],arm,i+1,len(seq),*diffs,*[run_["counts"].get(k,"") for k in keys]])
    tmp=out/"bundle.tmp"
    with ZipFile(tmp,"w",ZIP_DEFLATED,compresslevel=3) as z:
        z.writestr("manifest.json",json.dumps(manifest,indent=1))
        z.writestr("summary.json",json.dumps(summary,indent=1))
        z.writestr("per_prompt.csv",buff.getvalue())
        for r in rows:
            z.writestr("rows/"+r["prompt"].replace(":","_")+".json",json.dumps(r,separators=(",",":")))
    tmp.replace(out/ARCHIVE)
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--revision",required=True)
    ap.add_argument("--per-domain",type=int,choices=range(1,51),default=50)
    ap.add_argument("--max-new",type=int,choices=range(1,257),default=256)
    args=ap.parse_args()
    if torch.cuda.device_count()!=2 or not all("T4" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Select Kaggle GPU T4 ×2")
    data,_=inputs(args.input)
    lock=json.loads((ROOT/"source-lock.json").read_text())
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    paths={m:snapshot_download(m,revision=lock["models"][m],allow_patterns=["*.json","*.safetensors","*.txt","*.model"])
           for m in (AR,TARGET)}
    tok,identity=tokenizers(paths)
    prompts=plan(tok,lock["datasets"],args.per_domain,{r["id"] for r in data["corpus"]+data["prompts"]})
    corpus=Index()
    for row in data["corpus"]:
        corpus.append([],row["token_ids"])
    for ctx in tuple(corpus.table):
        corpus.hits(ctx)
    q=load_model(paths[AR],"cuda:1")
    p,_,_=load_target(paths[TARGET],dict(target_precision="fp16",target_gpu0_layers=18,exclude_inter_gpu_transfer_time=False))
    eos=set(data["eos"])
    manifest=dict(revision=args.revision,models={m:lock["models"][m] for m in paths},datasets=lock["datasets"],
                  input_sha256=HASHES,prompts=prompts,arms=list(ARMS),max_new=args.max_new,eos=sorted(eos),
                  devices=[torch.cuda.get_device_name(i) for i in range(2)],torch=str(torch.__version__),
                  cuda=torch.version.cuda,placement=p.hf_device_map,vocab_sha256=identity["shared_vocab_sha256"],
                  dtype="float16",draft_length=DRAFT,tree_width=WIDTH,fork_limit=FORK_LIMIT,feedback_weight=P_WEIGHT,
                  old_p_min=P_MIN,q_topk=TOP_Q,fork_threshold=FORK_THRESHOLD,switch_threshold=SWITCH_THRESHOLD)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    meta=out/"manifest.json"
    if meta.exists() and json.loads(meta.read_bytes())!=manifest:
        raise ValueError("Checkpoint belongs to a different run")
    atomic(meta,manifest)
    signature=sha256(meta.read_bytes()).hexdigest()
    with torch.inference_mode(),threadpool_limits(limits=2):
        warm=tok.apply_chat_template([{"role":"user","content":"What is 5 + 7?"}],tokenize=True,add_generation_prompt=True)
        for arm in ARMS:
            if arm=="greedy7b":
                reference(p,warm,eos,12)
            else:
                run(q,p,warm,corpus,eos,24,arm)
        rows=[]
        for i,item in enumerate(prompts):
            pid=item["id"]
            order=list(ARMS)
            Random("consensus-tree-v1/"+pid).shuffle(order)
            row=dict(prompt=pid,dataset=item["dataset"],order=order,runs={})
            for arm in order:
                file=out/(pid.replace(":","_")+"_"+arm+".json")
                if file.exists():
                    saved=json.loads(file.read_bytes())
                    if (saved["signature"],saved["prompt"],saved["arm"])!=(signature,pid,arm):
                        raise ValueError(f"Checkpoint mismatch: {file}")
                    value=saved["run"]
                else:
                    prompt=item["prompt_tokens"]
                    value=(reference(p,prompt,eos,args.max_new) if arm=="greedy7b"
                           else run(q,p,prompt,corpus,eos,args.max_new,arm))
                    if not 0<len(value["tokens"])<=args.max_new or not value["counts"]["rounds"]:
                        raise ValueError(f"Incomplete output: {pid} {arm}")
                    atomic(file,dict(signature=signature,prompt=pid,arm=arm,run=value))
                    print(f"{i+1}/{len(prompts)} {pid} {arm}: {value['counts']['wall_ms']/1000:.2f}s",flush=True)
                row["runs"][arm]=value
            rows.append(row)
            if len(rows)==1 or len(rows)%10==0 or len(rows)==len(prompts):
                summary=write_report(out,manifest,rows)
                print(f"Report: {len(rows)}/{len(prompts)} prompts",flush=True)
    print(json.dumps(summary["all"]["pairs"],indent=1))
    print(out/ARCHIVE)

if __name__=="__main__":
    main()
