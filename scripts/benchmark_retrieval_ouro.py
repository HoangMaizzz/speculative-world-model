"""Prospective paired 100 MATH-500 + 100 GSM8K comparison, pinned to six configurations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch
from huggingface_hub import snapshot_download
from threadpoolctl import threadpool_limits

from benchmark_retrieval_alignment import HASHES, inputs, rollout
from ouro_qwen import ouroboros_rollout
from retrieval_alignment_core import Index
from retrieval_ouro_report import ARMS, write_bundle
from suffix_lab.assets import ROOT, TARGET, AR
from suffix_lab.runtime import load_model, load_target, tokenizers

def atomic(path,data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,separators=(",",":")),encoding="utf8")
    tmp.replace(path)

def plan(tokenizer,datasets,per_domain,excluded):
    from datasets import load_dataset
    indices=json.loads((ROOT/"configs/retrieval_ouro_200.json").read_text())
    groups=[]
    for domain,name in (("math500","HuggingFaceH4/MATH-500"),("gsm8k","openai/gsm8k")):
        dataset=load_dataset(name,**({"name":"main"} if domain=="gsm8k" else {}),split="test",revision=datasets[name])
        group=[]
        if len(set(indices[domain]))!=100 or len(indices[domain])!=100:
            raise ValueError("Expected 100 distinct frozen IDs per domain")
        for i in indices[domain][:per_domain]:
            question=dataset[i]["question" if domain=="gsm8k" else "problem"]
            ids=tokenizer.apply_chat_template([{"role":"user","content":question}],tokenize=True,add_generation_prompt=True)
            pid=f"{domain}:{i}"
            if pid in excluded:
                raise ValueError(f"Historical prompt in evaluation plan: {pid}")
            group.append(dict(id=pid,dataset=domain,prompt_tokens=ids))
        groups.append(group)
    return [r for pair in zip(*groups,strict=True) for r in pair]

def run_arm(arm,q,p,ids,eos,limit,corpus,selector):
    if arm in ("O4","O16"):
        run=ouroboros_rollout(q,p,ids,eos,limit,int(arm[1:]))
        counts=dict(run["counts"])
        counts.update(rounds=counts["verify_rounds"],accepted=counts["accept"])
        return dict(tokens=run["token_ids"],counts=counts,rounds=run["rounds"])
    return rollout(q,p,ids,eos,limit,arm,corpus,selector)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--revision",required=True)
    ap.add_argument("--per-domain",type=int,choices=range(1,101),default=100)
    ap.add_argument("--max-new",type=int,choices=range(1,257),default=256)
    args=ap.parse_args()
    if torch.cuda.device_count()!=2 or not all("T4" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("This report benchmark requires two Tesla T4 GPUs")
    data,selector=inputs(args.input)
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
    manifest=dict(revision=args.revision,upstream="7dac65b7327a9f41acc741d57d1955a774beb34d",
                  models={m:lock["models"][m] for m in paths},datasets=lock["datasets"],input_sha256=HASHES,
                  prompts=prompts,arms=list(ARMS),max_new=args.max_new,eos=sorted(eos),dtype="float16",
                  devices=[torch.cuda.get_device_name(i) for i in range(2)],torch=str(torch.__version__),
                  cuda=torch.version.cuda,placement=p.hf_device_map,vocab_sha256=identity["shared_vocab_sha256"],
                  draft_lengths={"A":16,"B":16,"C":16,"D":16,"O16":16,"O4":4},
                  ouroboros=dict(level=7,window=20,topk=3,initial_pool="empty"),
                  threshold=selector["threshold"],corpus_size=len(data["corpus"]))
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
            run_arm(arm,q,p,warm,eos,32,corpus,selector)
        rows=[]
        for i,item in enumerate(prompts):
            pid=item["id"]
            order=list(ARMS)
            Random("report-200/"+pid).shuffle(order)
            row=dict(prompt=pid,dataset=item["dataset"],order=order,runs={})
            for arm in order:
                file=out/(pid.replace(":","_")+"_"+arm+".json")
                if file.exists():
                    saved=json.loads(file.read_bytes())
                    if (saved["signature"],saved["prompt"],saved["arm"])!=(signature,pid,arm):
                        raise ValueError(f"Checkpoint mismatch: {file}")
                    run=saved["run"]
                else:
                    run=run_arm(arm,q,p,item["prompt_tokens"],eos,args.max_new,corpus,selector)
                    if not 0<len(run["tokens"])<=args.max_new or not run["counts"]["rounds"]:
                        raise ValueError(f"Incomplete output: {pid} {arm}")
                    atomic(file,dict(signature=signature,prompt=pid,arm=arm,run=run))
                    print(f"{i+1}/{len(prompts)} {pid} {arm}: {run['counts']['wall_ms']/1000:.2f}s",flush=True)
                row["runs"][arm]=run
            rows.append(row)
            if len(rows)%10==0 or len(rows)==len(prompts):
                summary=write_bundle(out,manifest,rows)
                print(f"Report: {len(rows)}/{len(prompts)} prompts",flush=True)
    print(json.dumps(summary["all"]["pairs"]["B_vs_A"],indent=1))
    print(out/"retrieval_ouro_results.zip")

if __name__=="__main__":
    main()

