"""Check causal KV compaction, greedy parity and report output with tiny Qwen2 models."""
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from benchmark_consensus_tree import ARCHIVE, ARMS, reference, write_report
from consensus_tree_core import Feedback, MODES, Index, guided_decode, run
from retrieval_alignment_core import ids_tensor

def main():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    cfg=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                    num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True)
    q=Qwen2ForCausalLM(cfg).eval()
    p=Qwen2ForCausalLM(cfg).eval()
    corpus=Index()
    corpus.append([],[7]*40)
    rows=[]
    prompt=[3,6,7]
    with torch.inference_mode():
        for i,target in enumerate((q,p)):
            ref=reference(target,prompt,set(),37)
            runs={mode:run(q,target,prompt,corpus,set(),37,mode) for mode in MODES}
            assert all(r["tokens"]==ref["tokens"] for r in runs.values())
            assert all(0<r["counts"]["tree_nodes"]<=16*r["counts"]["draft_forwards"] for r in runs.values())
            rows.append(dict(prompt=f"toy:{i}",dataset=("math500","gsm8k")[i],order=list(ARMS),
                             runs={**runs,"greedy7b":ref}))
        # Force a correction candidate that Q can see in the very first packet.
        raw=Index()
        initial=q(ids_tensor(q,prompt),use_cache=True).logits[0,-1].float().softmax(-1)
        _, ids=initial.topk(2)
        a,b=ids.tolist()
        raw.append([6,7],[b,7,7])
        old_q={a:.99,b:.01}
        old_p={a:.005,b:.995}
        mem=Feedback([6,7],[a],[old_q],[old_p])
        for mode in ("fork","switch"):
            cache=q.model(ids_tensor(q,prompt[:-1]),use_cache=True).past_key_values
            draft,_,choice,branch=guided_decode(q,cache,prompt,4,set(),[(raw,1.),(mem,1.5)],
                                               [(raw,mem)],mode,Counter())
            assert choice and choice[0]==0 and choice[1]==b, (a,b,choice)
            if mode=="fork":
                assert draft[0]==a and branch[0]==b and len(branch)>1
                conditional=q(ids_tensor(q,prompt+[b]),use_cache=False).logits[0,-1].argmax().item()
                assert branch[1]==conditional
            else:
                assert draft[0]==b
                conditional=q(ids_tensor(q,prompt+[b]),use_cache=False).logits[0,-1].argmax().item()
                assert draft[1]==conditional
            assert cache.get_seq_length()==len(prompt)+len(draft)-1
    with TemporaryDirectory() as directory:
        out=Path(directory)
        summary=write_report(out,{"prompts":[1,2]},rows)
        assert summary["complete"] and summary["all"]["arms"]["switch"]["exact_vs_greedy7b"]==2
        with ZipFile(out/ARCHIVE) as z:
            assert z.testzip() is None and len(z.read("per_prompt.csv").splitlines())==11
            assert json.loads(z.read("summary.json"))["completed"]==2
    print("PASS: 8 full rollouts, forced fork and switch under a new causal prefix, and report ZIP.")

if __name__=="__main__":
    main()
