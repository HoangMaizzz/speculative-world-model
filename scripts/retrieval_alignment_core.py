"""Greedy retrieval packets, frozen alignment features and packed 7B verification."""
from collections import Counter, defaultdict
from heapq import heappop, heappush

import numpy as np
import torch
from benchmark_dllm_ar32 import CachedTarget
from suffix_lab.long60_plan import decide, until_eos


def longest_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


class Index:
    def __init__(self):
        self.table = defaultdict(Counter)
        self.cache = {}

    def append(self, left, ids):
        ctx = list(left[-4:])
        for token in ids:
            for n in range(1, len(ctx)+1):
                k = tuple(ctx[-n:])
                self.table[k][token] += 1
                self.cache.pop(k, None)
            ctx = (ctx+[token])[-4:]

    def hits(self, ctx):
        for n in range(min(4,len(ctx)), 0, -1):
            k = ctx[-n:]
            if k not in self.table:
                continue
            if k not in self.cache:
                counts = self.table[k]
                den = sum(counts.values()) + 1
                self.cache[k] = [(token, count/den) for token,count in counts.most_common(8)]
            return n, self.cache[k]
        return 0, ()


def tree(ctx, sources, budget, depth, other_mass=0.):
    nodes = set()
    todo = []
    def expand(path, mass):
        key = (ctx+path)[-4:]
        weights = Counter()
        for idx, prior in sources:
            n,hits = idx.hits(key)
            factor = prior * (4**n)
            for token, prob in hits:
                weights[token] += factor*prob
        total = sum(weights.values())
        if total:
            for token,w in weights.most_common(6):
                heappush(todo, (-mass*w/(total+other_mass), path+(token,)))
    if depth:
        expand((),1.)
    while todo and len(nodes)<budget:
        neg, path = heappop(todo)
        nodes.add(path)
        if len(path)<depth and len(nodes)<budget:
            expand(path,-neg*.85)
    return nodes


def decode(m, kv, ctx, limit, eos, sources, width, counts, collect=False):
    generated=[]
    winning_rows=[]
    while len(generated)<limit:
        remaining=limit-len(generated)
        query=ctx+generated
        paths=sorted(tree(tuple(query[-4:]),sources,width,remaining-1),key=lambda p:(len(p),p)) if width else []
        size=kv.get_seq_length()
        path_id={p:i+1 for i,p in enumerate(paths)}
        ids=[query[-1]]+[p[-1] for p in paths]
        args={}
        if paths:
            mask=torch.zeros((len(ids),size+len(ids)),dtype=m.dtype)
            mask[:,size:]=torch.finfo(m.dtype).min
            mask[:,size]=0
            for j,p in enumerate(paths,1):
                for k in range(1,len(p)+1):
                    mask[j,size+path_id[p[:k]]]=0
            args['attention_mask']=mask[None,None].to(m.device)
            args['position_ids']=torch.tensor([[size]+[size+len(p) for p in paths]],device=m.device)
        logits=m(input_ids=torch.tensor([ids],device=m.device),past_key_values=kv,
                    use_cache=True,**args).logits[0]
        predicted=logits.argmax(-1).tolist()
        if counts is not None:
            counts.update(draft_forwards=1,tree_forwards=int(bool(paths)),tree_nodes=len(paths))
        keep=[0]
        path=()
        idx=0
        finished=False
        while True:
            token=predicted[idx]
            generated.append(token)
            if collect:
                winning_rows.append(logits[idx])
            if token in eos or len(generated)==limit:
                finished=True
                break
            path=path+(token,)
            if path not in path_id:
                break
            idx=path_id[path]
            keep.append(idx)
        if paths:
            if keep!=list(range(len(keep))):
                indices=torch.tensor([size+j for j in keep],device=m.device)
                for cache in (kv.key_cache,kv.value_cache):
                    for layer in cache:
                        layer[:,:,size:size+len(keep)].copy_(layer.index_select(2,indices))
            kv.crop(size+len(keep))
        if finished:
            return generated, winning_rows
    return generated, winning_rows


SELECTORS = ("Q", "Q+raw+alignedP")
COLUMNS = (slice(0, 12), slice(None))


def pointers(query, left, suffix, position):
    text = left + suffix
    row = [0] * (len(text) + 1)
    for i, token in enumerate(query, 1):
        update = [i]
        for j, other in enumerate(text):
            update.append(min(row[j] + (token != other), row[j + 1] + 1, update[-1] + 1))
        row = update
    start = len(left)
    options = [
        (row[start + j], -longest_prefix(query[::-1], text[max(0, start + j - 5):start + j][::-1]),
         abs(position - j), j)
        for j in range(len(suffix))
    ]
    chosen = sorted(options)[:3]
    if position < len(options) and options[position] not in chosen:
        chosen.append(options[position])
    return chosen


def candidates(prefix, draft, q_ids, q_lp, entropy, previous, offset, eos, aligned):
    reject, left, suffix, probabilities = previous
    features, keys = [], []
    for position, token in enumerate(draft):
        ids = q_ids[position]
        lp = q_lp[position].astype(float)
        if ids[0] != token:
            continue
        survival = float(np.sum(q_lp[:position, 0].astype(float)))
        if aligned:
            query = (prefix + draft[:position])[-5:]
            matches = pointers(query, left, suffix, position)
        for rank, other in enumerate(ids[1:4], 2):
            other = int(other)
            if other in eos:
                continue
            row = [
                position, rank, lp[rank - 1], lp[0], lp[0] - lp[rank - 1],
                lp[0] - lp[1], survival, float(entropy[position]), reject,
                len(suffix), offset, other == prefix[-1],
            ]
            if aligned:
                found = [j for j, value in enumerate(suffix) if value == other]
                row += [
                    len(found), min((abs(position - j) for j in found), default=65),
                    max((longest_prefix(query[::-1], (left + suffix[:j])[-5:][::-1])
                         for j in found), default=0),
                    int(any(suffix[j] == other for _, _, _, j in matches)),
                ]
                values = []
                for edit, neg, distance, j in matches:
                    prob = probabilities[j].get(other, 0.)
                    values.append((prob, prob / ((1 + edit) * (1 + .12 * distance)), edit, -neg, distance))
                if values:
                    best = max(values, key=lambda item: item[1])
                    row += [best[0], best[1], sum(item[1] for item in values), *best[2:]]
                else:
                    row += [0., 0., 0., 6., 0., 65.]
            features.append(row)
            keys.append((position, other))
    width = 22 if aligned else 12
    return np.asarray(features, dtype=np.float32).reshape(-1, width), keys


def ids_tensor(model, tokens):
    return torch.tensor([tokens], device=model.device)


def generate(model, cache, last, count, eos, collect=0):
    tokens, logits = [], []
    for _ in range(count):
        row = model(ids_tensor(model, [last]), past_key_values=cache, use_cache=True,
                    logits_to_keep=1).logits[0, -1]
        last = int(row.argmax())
        tokens.append(last)
        if len(tokens) <= collect:
            logits.append(row)
        if last in eos:
            break
    return tokens, logits


def select(rows, model, prefix, draft, previous, offset, eos, aligned, always):
    logits = torch.stack(rows).float()
    logp = logits.log_softmax(dim=-1)
    top_lp, top_ids = logp.topk(4, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    features, keys = candidates(prefix, draft, top_ids.cpu().numpy(), top_lp.cpu().numpy(),
                                entropy.cpu().numpy(), previous, offset, eos, aligned)
    if not keys:
        return None
    column = COLUMNS[int(aligned)]
    scores = model["classifier"].predict_proba(features[:, column])[:, 1]
    best = int(scores.argmax())
    if not always and scores[best] < model["threshold"]:
        return None
    position, token = keys[best]
    return position, token, float(scores[best])


class LiveTarget(CachedTarget):
    def check(self, prefix, draft, branch, position, eos, remaining, collect):
        """Score both paths with ancestor attention and compact the chosen KV."""
        model = self.model
        base = self.cache.get_seq_length()
        pending = len(self.pending)
        chain_end = pending + len(draft)
        query = self.pending + draft + branch
        settings = {}
        if branch:
            size = len(query)
            blocked = torch.ones((size, size), dtype=torch.bool, device=model.device).triu(1)
            blocked[chain_end:, pending + position:chain_end] = True
            mask = torch.zeros((1, 1, size, base + size), dtype=model.dtype, device=model.device)
            mask[..., base:].masked_fill_(blocked, torch.finfo(model.dtype).min)
            branch_begin = base + pending + position
            positions = list(range(base, base + chain_end)) + list(range(branch_begin, branch_begin + len(branch)))
            settings = dict(attention_mask=mask, position_ids=ids_tensor(model, positions))
        output = model(ids_tensor(model, query), past_key_values=self.cache, use_cache=True,
                       logits_to_keep=len(query), **settings)
        self.cache = output.past_key_values
        logits = output.logits[0]
        predictions = logits.argmax(-1).tolist()
        shifted = predictions[pending - 1:chain_end] if pending else [self.next_token] + predictions[:chain_end]
        decision = decide(draft, shifted, eos, remaining)
        rejected = decision["first_rejection"]
        baseline = decision["committed_tokens"]
        emitted, accepted = baseline, min(len(baseline), decision["accepted_draft"])
        matched = 0
        if branch and rejected == position and branch[0] == shifted[position]:
            after = predictions[chain_end:]
            matched = longest_prefix(branch, [shifted[position]] + after)
            extension = branch[:matched]
            if extension[-1] not in eos:
                extension.append(after[matched - 1])
            emitted = until_eos(draft[:position] + extension, eos)[:remaining]
            matched = min(matched, len(emitted) - position)
            accepted = position + matched
            keep_start = base + pending + position
            source = base + chain_end
            for states in (self.cache.key_cache, self.cache.value_cache):
                for tensor in states:
                    tensor[..., keep_start:keep_start + matched, :].copy_(
                        tensor[..., source:source + matched, :].clone())

        previous = None
        if rejected is not None and not matched:
            suffix = until_eos(draft[rejected + 1:], eos, include=False)
            left = (prefix + draft[:rejected + 1])[-5:]
            probs = []
            if collect and suffix:
                start = pending + rejected
                rows = logits[start:start + len(suffix)].float()
                _, top_ids = rows.topk(32, dim=-1)
                top_probs = rows.softmax(dim=-1).gather(1, top_ids)
                probs = [dict(zip(i, p)) for i, p in zip(top_ids.tolist(), top_probs.tolist(), strict=True)]
            previous = (rejected, left, suffix, probs)

        self.cache.crop(base + pending + accepted)
        self.pending = emitted[accepted:]
        return emitted, accepted, matched, len(emitted) - len(baseline), previous
