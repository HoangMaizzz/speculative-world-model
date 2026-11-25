"""Exercise real attention masks, KV cropping, and the paired report using small CPU Qwen2 models."""
import argparse
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from benchmark_dllm_ar32 import decode_ar
from benchmark_retrieval_alignment import inputs
from benchmark_retrieval_ouro import run_arm
from retrieval_alignment_core import Index
from retrieval_ouro_report import ARMS,write_bundle

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    args=ap.parse_args()
    _,selector=inputs(args.input)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,
                    num_key_value_heads=2,tie_word_embeddings=True)
    q=Qwen2ForCausalLM(cfg).eval()
    p=Qwen2ForCausalLM(cfg).eval()
    idx=Index()
    idx.append([], [7]*40)
    rows=[]
    with torch.inference_mode():
        for i,target in enumerate([q,p]):
            prompt=[3,6,7]
            if i:
                # Untie this second target to test genuine mismatches with the drafter.
                torch.nn.init.normal_(target.lm_head.weight,std=.5)
                target.lm_head=torch.nn.Linear(32,64,bias=False)
            ref=decode_ar(target,prompt,set(),42)["token_ids"]
            runs={arm:run_arm(arm,q,target,prompt,set(),42,idx,selector) for arm in ARMS}
            for arm,run in runs.items():
                assert run["tokens"]==ref, (arm,i,run["tokens"],ref)
                assert run["counts"]["rounds"]==len(run["rounds"])
            if not i:
                assert runs["O4"]["counts"]["accept_phrase"]>0
            else:
                assert any(r["main_accepted"]<len(r["proposal"]) for r in runs["O16"]["rounds"])
            rows.append(dict(prompt=f"tiny:{i}",dataset=("math500","gsm8k")[i],order=list(ARMS),runs=runs))
    # Precision of reporting when a candidate diverges and the equal-output subset is empty.
    rows[0]["runs"]["D"]["tokens"]=[63]
    rows[1]["runs"]["D"]["tokens"]=[63]
    with tempfile.TemporaryDirectory() as tmp:
        s=write_bundle(Path(tmp),{"prompts":[1,2]},rows)
        assert s["complete"] and s["all"]["pairs"]["D_vs_A"]["same_output"] is None
        with ZipFile(Path(tmp)/"retrieval_ouro_results.zip") as z:
            assert z.testzip() is None and len(z.read("per_prompt.csv").splitlines())==13
            assert json.loads(z.read("summary.json"))["all"]["arms"]["B"]["exact_vs_A"]==2
    print("PASS: 12 tiny Qwen rollouts vs sequential target, phrase extension, rejection, ZIP and paired summaries.")

if __name__=="__main__":
    main()
