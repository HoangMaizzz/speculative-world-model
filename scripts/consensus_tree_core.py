"""Online retrieval with verifier feedback and one optional alternative per round."""
from collections import Counter, defaultdict, deque
from math import log
from time import perf_counter

import torch

from retrieval_alignment_core import Index, LiveTarget, decode, ids_tensor, longest_prefix, tree

MODES = ("base", "boost", "fork", "switch")
WIDTH = 16
DRAFT = 16
FORK_LIMIT = 4
FORK_THRESHOLD = .025
SWITCH_THRESHOLD = .4
P_MIN = .5
P_WEIGHT = 1.5
TOP_Q = 16

def sync():
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.synchronize(gpu)

class Feedback:
    def __init__(self,left,tail,q_rows,p_rows):
        self.records=defaultdict(list)
        self.index=defaultdict(Counter)
        for j,(token,q,p) in enumerate(zip(tail,q_rows,p_rows,strict=True)):
            q0=max(q,key=q.get)
            p0=max(p,key=p.get)
            if token!=q0 or q0==p0 or p[p0]<P_MIN:
                continue
            ctx=(left+tail[:j])[-4:]
            for n in range(1,len(ctx)+1):
                key=tuple(ctx[-n:])
                self.records[key].append((q,p))
                if n>=2:
                    self.index[key][p0]+=p[p0]

    def hits(self,ctx):
        for n in range(min(4,len(ctx)),1,-1):
            counts=self.index.get(ctx[-n:])
            if counts:
                total=sum(counts.values())+1.
                return n,[(t,w/total) for t,w in counts.most_common(8)]
        return 0,()

def prefer(ctx,q,q0,bank,allowed,eos,threshold):
    for n in range(min(4,len(ctx)),0,-1):
        key=tuple(ctx[-n:])
        for _,memory in reversed(bank):
            for old_q,old_p in memory.records.get(key,()):
                support={x for x in q.keys() & old_q.keys() & old_p.keys()
                         if q[x]>0 and old_q[x]>0 and old_p[x]>0}
                if q0 not in support:
                    continue
                def score(token):
                    corr=log(old_p[token]/old_q[token])
                    return log(q[token])+.5*max(-3.,min(3.,corr))
                x=max(sorted(support),key=score)
                if x != q0 and x not in eos and x in allowed:
                    gain=score(x)-score(q0)
                    if gain>threshold:
                        return x,gain
    return None

def compact(cache,size,keep):
    if keep != list(range(len(keep))):
        for states in (cache.key_cache,cache.value_cache):
            for layer in states:
                indices=torch.tensor([size+j for j in keep],device=layer.device)
                layer[:,:,size:size+len(keep)].copy_(layer.index_select(2,indices))
    cache.crop(size+len(keep))

def guided_decode(q,cache,ctx,limit,eos,sources,bank,mode,counts):
    draft=[]
    q_rows=[]
    branch=[]
    chosen=None
    while len(draft)<limit:
        query=ctx+draft
        paths=sorted(tree(tuple(query[-4:]),sources,WIDTH,limit-len(draft)-1),key=lambda p:(len(p),p))
        size=cache.get_seq_length()
        ids=[query[-1]]+[p[-1] for p in paths]
        locations={p:i+1 for i,p in enumerate(paths)}
        children=defaultdict(set)
        for path in paths:
            children[path[:-1]].add(path[-1])
        params={}
        if paths:
            mask=torch.full((len(ids),size+len(ids)),torch.finfo(q.dtype).min,dtype=q.dtype)
            mask[:,:size+1]=0
            for j,path in enumerate(paths,1):
                for k in range(1,len(path)+1):
                    mask[j,size+locations[path[:k]]]=0
            params['attention_mask']=mask[None,None].to(q.device)
            params['position_ids']=ids_tensor(q,[size]+[size+len(p) for p in paths])
        logits=q(ids_tensor(q,ids),past_key_values=cache,use_cache=True,**params).logits[0]
        vals,top=logits.float().log_softmax(-1).topk(TOP_Q,dim=-1)
        greedy=logits.argmax(-1).tolist()
        top=top.tolist()
        prob=[dict(zip(tokens,values)) for tokens,values in zip(top,vals.exp().tolist(),strict=True)]
        counts.update(draft_forwards=1,tree_forwards=bool(paths),tree_nodes=len(paths))
        keep=[0]
        path=()
        idx=0
        while True:
            token=greedy[idx]
            if mode in ("fork","switch") and chosen is None and bank and len(draft)<limit-1:
                suggestion=prefer(ctx+draft,prob[idx],token,bank,children[path],eos,
                                  SWITCH_THRESHOLD if mode=="switch" else FORK_THRESHOLD)
                if suggestion:
                    alt,margin=suggestion
                    chosen=(len(draft),alt,float(margin))
                    if mode=="switch":
                        token=alt
                    else:
                        branch=[alt]
                        alt_path=path+(alt,)
                        # Read continuation logits computed under the alternative's own causal ancestry.
                        while alt_path in locations and len(branch)<min(FORK_LIMIT,limit-len(draft)):
                            next_token=greedy[locations[alt_path]]
                            branch.append(next_token)
                            if next_token in eos:
                                break
                            alt_path+=(next_token,)
            q_rows.append(prob[idx])
            draft.append(token)
            if token in eos or len(draft)==limit:
                break
            path+=(token,)
            if path not in locations:
                break
            idx=locations[path]
            keep.append(idx)
        compact(cache,size,keep)
        if draft[-1] in eos:
            break
    return draft,q_rows,chosen,branch

def run(q,p,prompt,corpus,eos,limit,mode):
    prefix=list(prompt)
    history=Index()
    history.append([],prefix)
    bank=deque(maxlen=4)
    rounds=[]
    counts=Counter()
    sync()
    start=perf_counter()
    cache=q.model(ids_tensor(q,prefix[:-1]),use_cache=True).past_key_values
    verifier=LiveTarget(p,prefix)
    sync()
    now=perf_counter()
    counts["prefill_ms"]=1000*(now-start)
    while len(prefix)-len(prompt)<limit and prefix[-1] not in eos:
        offset=len(prefix)-len(prompt)
        remaining=limit-offset
        missing=prefix[cache.get_seq_length():]
        if len(missing)>1:
            q.model(ids_tensor(q,missing[:-1]),past_key_values=cache,use_cache=True)
            counts["q_fill_calls"]+=1
        sources=[(history,1.),(corpus,.75)]
        for age,(raw,mem) in enumerate(reversed(bank)):
            sources.append((raw,1.5*.75**age))
            if mode!="base":
                sources.append((mem,P_WEIGHT*.75**age))
        chosen=None
        branch=[]
        if mode=="base":
            d,_=decode(q,cache,prefix[-4:],min(DRAFT,remaining),eos,sources,WIDTH,counts)
            qrows=[]
        else:
            d,qrows,chosen,branch=guided_decode(q,cache,prefix[-4:],min(DRAFT,remaining),eos,sources,bank,mode,counts)
        assert cache.get_seq_length()==len(prefix)+len(d)-1
        sync()
        drafted=perf_counter()
        counts["draft_ms"]+=(drafted-now)*1000
        pos=chosen[0] if branch else 0
        if mode=="base":
            result=verifier.verify(d,eos,remaining)
            emitted=result["committed_tokens"]
            accepted=min(len(emitted),result["accepted_draft"])
            matched,extra,previous=0,0,None
        else:
            emitted,accepted,matched,extra,previous=verifier.check(prefix,d,branch,pos,eos,remaining,True)
        sync()
        checked=perf_counter()
        counts["verify_ms"]+=(checked-drafted)*1000
        if matched:
            cache.crop(len(prefix)+pos)
        cache.crop(min(cache.get_seq_length(),len(prefix)+accepted))
        r=longest_prefix(d,emitted)
        if r<min(len(d),len(emitted)):
            suffix=d[r+1:]
            suffix=suffix[:next((k for k,x in enumerate(suffix) if x in eos),len(suffix))]
            if suffix:
                left=(prefix+d[:r+1])[-4:]
                raw=Index()
                raw.append(left,suffix)
                mem=Feedback(left,suffix,qrows[r+1:r+1+len(suffix)],previous[3]) if previous else Feedback([],[],[],[])
                bank.append((raw,mem))
                counts["feedback_rows"]+=sum(len(v) for k,v in mem.records.items() if len(k)==1)
        history.append(prefix[-4:],emitted)
        prefix.extend(emitted)
        assert verifier.cache.get_seq_length()+len(verifier.pending)==len(prefix)
        hit=bool(chosen and (matched if mode=="fork" else chosen[0]<accepted))
        if not emitted:
            raise AssertionError("Verification failed to advance")
        counts.update(rounds=1,accepted=accepted,actions=chosen is not None,hits=hit,extra=extra,
                      verifier_branch_nodes=len(branch))
        rounds.append(dict(offset=offset,draft=d,branch=branch,action=chosen,hit=hit,accepted=accepted,emitted=emitted))
        now=perf_counter()
        counts["maintenance_ms"]+=(now-checked)*1000
    sync()
    counts["wall_ms"]=(perf_counter()-start)*1000
    return dict(tokens=prefix[len(prompt):],counts=dict(counts),rounds=rounds)
