"""Frozen-verifier source-span pilot. No target weights or future features."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile


def sha(b):
    return hashlib.sha256(b).hexdigest()


PINS = {
    "events.jsonl": "d29341f11d901efdd29e35ba80abdc7c216ec551547b35ada6fd6f5b934a0c29",
    "offline_labels.parquet": "9cbe44001fd4a5db4196accb94f77615d33eed2235651a5cf4a22fc3fe953315",
    "shards-index.json": "257ce630dd2368e66e156272fd0a3e7e03cbe2e2fa00c3041202795d3e0c84bf",
    "replay_checks.jsonl": "d63104c9e8c09664e93fc4efb0cb5949534d16ac459e088d48d6ad97a4dfe490",
}
FOLDS = ("train", "validation", "test")
ARMS = {"scalar": (False, False), "scalar_corr": (False, True),
        "hidden": (True, False), "hidden_corr": (True, True)}


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf8")
    temp.replace(path)


def prepare(root, output, smoke):
    import numpy as np
    import pyarrow.parquet as pq
    from safetensors.numpy import load
    cache = output/"features.npz"
    events_file = output/"events.json"
    if cache.exists() and events_file.exists():
        with np.load(cache) as z:
            return json.loads(events_file.read_text()), z["scalar"], z["hidden"]
    with ExitStack() as stack:
        if root.is_file():
            archive = stack.enter_context(zipfile.ZipFile(root))
            read = archive.read
        elif (root/"events.jsonl").is_file():
            read = lambda name: (root/name).read_bytes()
        else:
            parts = sorted(root.rglob("verifier_features_part-*.zip"))
            if not parts:
                raise FileNotFoundError("Point INPUT to the original feature ZIP, extracted archive, or all split ZIPs")
            locations = {}
            for part in parts:
                a = stack.enter_context(zipfile.ZipFile(part))
                for name in a.namelist():
                    if not name.endswith("/"):
                        if name in locations:
                            raise ValueError(f"Duplicate split member: {name}")
                        locations[name] = a
            read = lambda name: locations[name].read(name)
        metadata = {n:read(n) for n in PINS}
        for name, data in metadata.items():
            if sha(data) != PINS[name]:
                raise ValueError(f"Checkpoint hash differs: {name}")
        events = {r["id"]:r for r in map(json.loads,metadata["events.jsonl"].splitlines())}
        labels = {r["id"]:r for r in pq.read_table(io.BytesIO(metadata["offline_labels.parquet"])).to_pylist()}
        checks = {r["id"]:r for r in map(json.loads,metadata["replay_checks.jsonl"].splitlines())}
        folds = {}
        for e in events.values():
            if folds.setdefault(e["prompt_id"],e["prompt_split"]) != e["prompt_split"]:
                raise ValueError("Prompt leakage")
        assert set(events)==set(labels)==set(checks)
        assert Counter(e["prompt_split"] for e in events.values()) == dict(zip(FOLDS,(3131,359,357)))
        assert Counter(folds.values()) == dict(zip(FOLDS,(164,19,17)))
        assert all(c["old_top1_matches"] for c in checks.values())
        parts=json.loads(metadata["shards-index.json"])["parts"]
        if smoke:
            eligible=[e for e in events.values() if e["correction_emitted_within_output_budget"] and len(e["source_ids"])>=8 and len(labels[e["id"]]["reference_ids"])>=8]
            chosen=set()
            for fold in FOLDS:
                group=[e for e in eligible if e["prompt_split"]==fold]
                # Fixed spread across the archive, independent of outcomes.
                chosen.update(group[i*len(group)//24]["id"] for i in range(24))
        else:
            chosen={e["id"] for e in events.values() if e["correction_emitted_within_output_budget"]}
        size=sum(len(events[k]["source_ids"]) for k in chosen)
        hidden=np.lib.format.open_memmap(output/"layer27.npy",mode="w+",dtype="float16",shape=(size,3584))
        scalar, result=[], []
        offset=0
        seen=set()
        for p in parts:
            name=p["path"]
            ids=[k for k in chosen if events[k]["tensors"]["shard"]==name]
            if not ids:
                continue
            raw=read(name)
            if sha(raw)!=p["sha256"]:
                raise ValueError(f"Shard hash differs: {name}")
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for key in sorted(ids):
                    e=events[key]; label=labels[key]; t=e["tensors"]
                    payload=z.read(t["member"])
                    assert sha(payload)==t["sha256"] and key not in seen
                    seen.add(key)
                    ts=load(payload)
                    src=e["source_ids"]; a=e["source_proposal_start"]; n=len(src)
                    assert ts["source_ids"].tolist()==src==e["proposal_ids"][a:a+n]
                    assert a==e["accepted"]+1
                    assert ts["hidden_absolute_positions"].tolist()==list(range(e["source_absolute_start"]-1,e["source_absolute_start"]+n))
                    h=ts["old_layer_27"]
                    assert h.shape==(n+1,3584) and h.dtype==np.float16 and np.isfinite(h).all()
                    hidden[offset:offset+n]=h[1:]
                    old=e["old"]; draft=e["drafter"]
                    for s in draft["trace"]:
                        assert draft["order"][s["position"]]==s["step"]
                    f=np.column_stack([np.arange(n)/64, np.full(n,n/64), np.full(n,a/64),
                       np.full(n,e["offset"]/256),np.full(n,e["dataset"]=="math500"),
                       draft["confidence"][a:a+n], np.asarray(draft["order"][a:a+n])/64,
                       *[old[k][a:a+n] for k in ("verifier_logp_draft","verifier_greedy_match",
                         "verifier_gap_to_top1","entropy","logit_margin","top1_probability")],
                       np.log1p(old["verifier_rank_draft"][a:a+n]),
                       [src.count(t)/n for t in src],
                       np.log1p(np.linalg.norm(h[1:].astype("float32"),axis=1))]).astype("float32")
                    assert np.isfinite(f).all()
                    ref=label["reference_ids"]
                    for run in label["runs"]:
                        i,j,k=run["source_start"],run["destination_start"],run["length"]
                        assert src[i:i+k]==ref[j:j+k] and run["delta"]==j-i
                    scalar.append(f)
                    result.append(dict(id=key,prompt=e["prompt_id"],fold=e["prompt_split"],dataset=e["dataset"],
                        offset=offset,src=src,ref=ref,correction=e["correction"],rejected=e["proposal_ids"][a-1]))
                    offset+=n
            print(f"Read {offset}/{size} source tokens",flush=True)
        assert seen==chosen and offset==size
        train=np.concatenate([np.arange(e["offset"],e["offset"]+len(e["src"])) for e in result if e["fold"]=="train"])
        from sklearn.decomposition import PCA
        rng=np.random.default_rng(42)
        n_components=32
        def normalized(x):
            a=x.astype("float32")
            return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8)
        pca=PCA(n_components=n_components,svd_solver="randomized",random_state=42)
        pca.fit(normalized(hidden[rng.choice(train,min(len(train),4096),replace=False)]))
        reduced=np.empty((size,n_components),dtype="float32")
        for i in range(0,size,2048):
            reduced[i:i+2048]=pca.transform(normalized(hidden[i:i+2048]))
        np.savez(output/"projection.npz",mean=pca.mean_,components=pca.components_)
        sc=np.concatenate(scalar)
        np.savez(output/"features.npz",scalar=sc,hidden=reduced)
        save(events_file,result)
        save(output/"integrity.json",dict(events=len(result),all_events=len(events),selected_tokens=size,
             by_fold=dict(Counter(e["fold"] for e in result)),full=not smoke,primary_features="old_path_only",
             corrected_replay_mismatches=sum(not c["corrected_next_matches_reference"] for c in checks.values())))
        del hidden
        (output/"layer27.npy").unlink()
        return result,sc,reduced


def windows(events, scalar, hidden, length, target):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    xs, hs, ids, ys, groups, starts=[],[],[],[],[],[]
    vocabulary={t:i+1 for i,t in enumerate(sorted({e[k] for e in events if e["fold"]=="train" for k in ("rejected","correction")}))}
    for group,e in enumerate(events):
        src,ref=e["src"],e["ref"]
        if min(len(src),len(ref))<length:
            continue
        destinations={}
        for j in range(min(len(ref)-length+1,8 if target=="near" else len(ref))):
            destinations.setdefault(tuple(ref[j:j+length]),[]).append(j)
        for i in range(len(src)-length+1):
            v=slice(e["offset"]+i,e["offset"]+i+length)
            x=scalar[v]
            xs.append(np.r_[x.mean(0),x.min(0),x.max(0),x[0],x[-1]])
            hs.append(hidden[v].ravel())
            ids.append([vocabulary.get(e["rejected"],0),vocabulary.get(e["correction"],0)])
            ys.append(any(j!=i for j in destinations.get(tuple(src[i:i+length]),())))
            groups.append(group); starts.append(i)
    X,H,C,Y,G,I=map(np.asarray,(xs,hs,ids,ys,groups,starts))
    masks={f:np.asarray([events[g]["fold"]==f for g in G]) for f in FOLDS}
    assert all(m.any() for m in masks.values()) and len(set(Y[masks["train"]]))==2
    sx,sh=(StandardScaler().fit(a[masks["train"]]) for a in (X,H))
    return sx.transform(X),sh.transform(H),C,Y,G,I,masks,vocabulary,sx,sh


def score_policy(scores, Y, G, I, masks, events, length):
    import numpy as np
    def picks(fold):
        indices=np.flatnonzero(masks[fold])
        ordered=sorted(indices,key=lambda i:(G[i],-scores[i],I[i]))
        selected={}; labels={}; top=Counter(); hits=set()
        for i in ordered:
            selected.setdefault(G[i],i)
            labels[G[i]]=labels.get(G[i],False) or bool(Y[i])
            top[G[i]]+=1
            if top[G[i]]<=8 and Y[i]: hits.add(G[i])
        return list(selected.values()),labels,len(hits)
    v,_,_=picks("validation")
    t,available,hit8=picks("test")
    def measure(indices, cut, fold):
        active=[i for i in indices if scores[i]>=cut]
        positives=sum(bool(Y[i]) for i in active)
        return dict(n=len(indices),selected=len(active),hits=positives,precision=positives/len(active) if active else None,
                    coverage=len(active)/len(indices),prompts=len({events[G[i]]["prompt"] for i in active}),
                    rows=[dict(event=events[G[i]]["id"],i=int(I[i]),score=float(scores[i]),correct=bool(Y[i])) for i in active])
    cuts={f"coverage_{q}":float(np.quantile(scores[v],1-q)) for q in (.02,.05,.1,.2,.5)}
    for p in (.9,.95):
        for i in sorted(v,key=lambda i:-scores[i]):
            m=measure(v,scores[i],"validation")
            if m["selected"]>=12 and m["precision"]>=p:
                cuts[f"validation_precision_{p}"]=float(scores[i])
    out={}
    for name,cut in cuts.items():
        v_metrics=measure(v,cut,"validation")
        t_metrics=measure(t,cut,"test")
        t_metrics.update(recovered_source_tokens=t_metrics["hits"]*length,
                         event_recall=t_metrics["hits"]/max(1,sum(available.values())),
                         shortlist_recall_at_8=hit8/max(1,sum(available.values())),
                         oracle_positive_events=sum(available.values()),
                         source_fraction=t_metrics["hits"]*length/sum(len(events[G[i]]["src"]) for i in t))
        t_metrics["domain"]={d:{k:v for k,v in measure([i for i in t if events[G[i]]["dataset"]==d],cut,"test").items()
                                if k!="rows"} for d in ("math500","gsm8k")
                             if any(events[G[i]]["dataset"]==d for i in t)}
        rng=np.random.default_rng(42)
        prompts=sorted({events[G[i]]["prompt"] for i in t})
        active=[i for i in t if scores[i]>=cut]
        counts=np.asarray([[sum(events[G[i]]["prompt"]==p for i in active),
                            sum(events[G[i]]["prompt"]==p and Y[i] for i in active)] for p in prompts])
        draw=counts[rng.integers(0,len(prompts),size=(2000,len(prompts)))].sum(1)
        ok=draw[:,0]>0
        t_metrics["prompt_bootstrap_precision_95"]=np.quantile(draw[ok,1]/draw[ok,0],[.025,.975]).tolist() if ok.any() else None
        out[name]=dict(threshold=cut,validation=v_metrics,test=t_metrics)
    return out


def train(events, scalar, hidden, output, smoke):
    import numpy as np
    import torch
    from torch import nn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    import joblib
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for length in ((4,) if smoke else (4,8)):
        for target in (("near",) if smoke else ("near","any")):
            X,H,C,Y,G,I,masks,vocab,sx,sh=windows(events,scalar,hidden,length,target)
            a,b,c=(np.flatnonzero(masks[f]) for f in FOLDS)
            np.savez(output/f"input_{target}_{length}.npz",groups=G,starts=I,labels=Y,train=a,validation=b,test=c)
            save(output/f"vocab_{target}_{length}.json",vocab)
            joblib.dump((sx,sh),output/f"scalers_{target}_{length}.joblib")
            tx,th,tc,ty=(torch.as_tensor(z,device=device,dtype=d) for z,d in
                         ((X,torch.float32),(H,torch.float32),(C,torch.long),(Y,torch.float32)))
            class Head(nn.Module):
                def __init__(self,h,c):
                    super().__init__()
                    self.h,self.c=h,c
                    self.embedding=nn.Embedding(len(vocab)+1,16,padding_idx=0) if c else None
                    self.net=nn.Sequential(nn.Linear(X.shape[1]+h*H.shape[1]+c*32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
                def forward(self,i):
                    z=[tx[i]]
                    if self.h: z.append(th[i])
                    if self.c: z.append(self.embedding(tc[i]).flatten(1))
                    return self.net(torch.cat(z,1)).squeeze(1)
            fits=[("hgb",42)]+[(name,seed) for name in ARMS for seed in ((42,) if smoke else (42,123,2026))]
            for name,seed in fits:
                stem=f"{target}_{length}_{name}_{seed}"
                marker=output/(stem+".json")
                if marker.exists(): continue
                started=time.perf_counter()
                if name=="hgb":
                    model=HistGradientBoostingClassifier(max_iter=12 if smoke else 90,max_leaf_nodes=15,random_state=seed)
                    model.fit(X[a],Y[a])
                    scores=model.predict_proba(X)[:,1]
                    joblib.dump(model,output/(stem+".joblib"))
                else:
                    torch.manual_seed(seed)
                    model=Head(*ARMS[name]).to(device)
                    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                    rng=np.random.default_rng(seed)
                    best=-1
                    for epoch in range(2 if smoke else 12):
                        model.train()
                        perm=rng.permutation(a)
                        for i in range(0,len(perm),1024):
                            q=perm[i:i+1024]
                            opt.zero_grad()
                            loss=nn.functional.binary_cross_entropy_with_logits(model(q),ty[q])
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(),1.)
                            opt.step()
                        model.eval()
                        with torch.inference_mode():
                            v=model(b).sigmoid().cpu().numpy()
                        ap=average_precision_score(Y[b],v)
                        if ap>best:
                            best=ap
                            state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
                    model.load_state_dict(state)
                    torch.save(state,output/(stem+".pt"))
                    with torch.inference_mode():
                        scores=np.concatenate([model(q).sigmoid().cpu().numpy() for q in np.array_split(np.arange(len(X)), max(1,(len(X)+8191)//8192))])
                np.savez(output/(stem+"_scores.npz"),validation=scores[b],test=scores[c])
                save(marker,dict(arm=name,seed=seed,length=length,target=target,seconds=time.perf_counter()-started,
                     validation_ap=float(average_precision_score(Y[b],scores[b])),
                     test_ap=float(average_precision_score(Y[c],scores[c])),
                     policy=score_policy(scores,Y,G,I,masks,events,length)))
                print("Completed",stem,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    identity=dict(script=sha(Path(__file__).read_bytes()),pins=PINS,smoke=a.smoke)
    identity_path=a.output/"identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError("Output identity differs. Use a new folder.")
    save(identity_path,identity)
    ev,x,h=prepare(a.input,a.output,a.smoke)
    train(ev,x,h,a.output,a.smoke)
    results=[json.loads(p.read_text()) for p in sorted(a.output.glob("*_*.json")) if p.stem.rsplit("_",1)[-1].isdigit() and not p.name.startswith("vocab_")]
    save(a.output/"summary.json",results)
    save(a.output/"status.json",dict(status="complete",fits=len(results),events=len(ev)))


if __name__=="__main__":
    main()
