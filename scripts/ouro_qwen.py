"""Qwen adapter for thunlp/Ouroboros at 7dac65b7327a9f41acc741d57d1955a774beb34d.
The two upstream modules and their Apache license are in ouroboros_reference/.
"""
from collections import Counter
from random import Random
from time import perf_counter

import torch
from transformers import DynamicCache
from ouroboros_reference import mask_making_llama as masks
from ouroboros_reference.cache_engine import CacheEngine
from retrieval_alignment_core import longest_prefix as lcp

LEVEL, WINDOW, TOPK = 7, 20, 3
PHRASE = LEVEL - 1

def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)

def flatten(paths):
    return [t for path in paths for t in path]


def compact(cache, base, branch_start, matched):
    if matched:
        for states in (cache.key_cache, cache.value_cache):
            for state in states:
                state[:, :, base:base + matched].copy_(
                    state[:, :, branch_start:branch_start + matched].clone())
    cache.crop(base + matched)


def pass_model(model, cache, ids, positions, mask, keep=0):
    device = model.device
    result = model(torch.tensor([ids], device=device), past_key_values=cache, use_cache=True,
                   position_ids=torch.tensor([positions], device=device),
                   attention_mask=mask.to(device), logits_to_keep=keep)
    return result.logits[0].argmax(-1).tolist()


class LookaheadDraft:
    def __init__(self, model, bank, prompt):
        self.model = model
        self.bank = bank
        self.cache = DynamicCache()
        rng = Random(10)
        self.levels = [[rng.choice(prompt) for _ in range(WINDOW + LEVEL - 3)]]

    def generate(self, prefix, limit, eos, counts):
        seq = list(prefix)
        end = len(seq) + limit
        while len(seq) < end:
            cache_size = self.cache.get_seq_length()
            pending = seq[cache_size:]
            key = seq[-1]
            fill = len(self.levels) - 1
            full = len(self.levels) == PHRASE
            guesses = self.bank.get_guess_tokens(key) if full and self.bank.has(key) else []
            guess = flatten(guesses)
            ids = pending + flatten(self.levels) + guess
            positions = list(range(cache_size, len(seq)))
            for i, level in enumerate(self.levels):
                begin = len(seq) - 1 + max(1, i)
                positions.extend(range(begin, begin + len(level)))
            positions.extend(list(range(len(seq), len(seq) + PHRASE)) * len(guesses))
            mask = masks.j_make_causal_mask_multilevel(
                [len(level) for level in self.levels], fill == 0, WINDOW, guess or None,
                PHRASE, False, False, (1, len(ids)), self.model.dtype, torch.device("cpu"),
                past_key_values_length=cache_size, extra_input_length=len(pending) - 1 if fill else 0)
            # Only the last pending token and subsequent rows are read by the algorithm.
            skip = len(pending) - 1
            pred = pass_model(self.model, self.cache, ids, positions, mask, keep=len(ids) - skip)
            lookahead_begin = len(ids) - skip - len(guess) - len(self.levels[-1])
            lookahead = pred[lookahead_begin:lookahead_begin + len(self.levels[-1])]
            counts.update(draft_forwards=1, draft_query_tokens=len(ids), draft_phrase_nodes=len(guess))
            next_token = pred[0]
            advance = [next_token]
            branch_start = 0
            for i, phrase in enumerate(guesses):
                start = 1 + len(flatten(self.levels)) + i * PHRASE
                follow = [next_token] + pred[start:start + PHRASE]
                # The public greedy code counts the first mismatch or the final phrase slot.
                length = min(lcp(phrase, follow) + 1, PHRASE, end - len(seq))
                if length > len(advance):
                    advance = follow[:length]
                    branch_start = cache_size + skip + start

            if fill == 0:
                self.levels = [self.levels[0][1:], lookahead]
            elif not full:
                self.levels = [level[1:] for level in self.levels] + [lookahead[1:]]
            else:
                self.bank.insert(key, lookahead, self.levels, WINDOW, LEVEL, WINDOW)
                self.levels = [self.levels[1][1:]] + self.levels[2:] + [lookahead]

            advance = truncate(advance, eos)
            compact(self.cache, len(seq), branch_start, len(advance) - 1)
            seq.extend(advance)
            if seq[-1] in eos:
                break
        proposal = seq[len(prefix):]
        guesses = self.bank.get_guess_tokens(seq[-1])[-TOPK:] if self.bank.has(seq[-1]) else []
        return proposal, guesses


def truncate(tokens, eos):
    size = next((i + 1 for i, token in enumerate(tokens) if token in eos), len(tokens))
    return tokens[:size]


def verify(target, cache, prefix, draft, phrases, eos, remaining):
    base = len(prefix) + len(draft)
    start = cache.get_seq_length()
    # At least one token immediately before the proposal is needed to predict its first token.
    assert start < len(prefix)
    shared = prefix[start:] + draft
    guess = flatten(phrases)
    ids = shared + guess
    positions = list(range(start, base)) + list(range(base, base + PHRASE)) * len(phrases)
    mask = masks.j_make_causal_mask_with_guess(
        len(guess), PHRASE, (1, len(ids)), target.dtype, torch.device("cpu"), start)
    # The target is never queried for an earlier prompt row.
    skip = len(prefix) - start - 1
    pred = pass_model(target, cache, ids, positions, mask, keep=len(ids) - skip)
    accepted = lcp(draft, pred)
    root_follow = pred[len(draft)]
    guesses_start = len(draft) + 1
    follow = [[root_follow] + pred[guesses_start + i * PHRASE:guesses_start + (i + 1) * PHRASE]
              for i in range(len(phrases))]
    if accepted < len(draft):
        emitted = truncate((draft[:accepted] + [pred[accepted]])[:remaining], eos)
        branch_accepted, branch_id = 0, None
        corrections = [tuple(row[:PHRASE]) for row in follow]
    elif draft[-1] in eos or len(draft) == remaining:
        emitted = draft
        branch_accepted, branch_id = 0, None
        corrections = [tuple(row[:PHRASE]) for row in follow]
    else:
        matched = [min(lcp(p, y), PHRASE - 1) for p, y in zip(phrases, follow)]
        branch_id = max(range(len(matched)), key=matched.__getitem__) if max(matched, default=0) else None
        extension = follow[branch_id][:matched[branch_id] + 1] if branch_id is not None else [root_follow]
        emitted = truncate((draft + extension)[:remaining], eos)
        branch_accepted = min(max(matched, default=0), max(0, len(emitted) - len(draft)))
        corrections = [tuple(row[:PHRASE]) for row in follow]
    consumed = min(accepted, len(emitted) - 1) if accepted == len(emitted) else accepted
    # Keep one final shared token pending when output stops on a matched EOS or at the cap.
    retained_phrases = min(branch_accepted, max(0, len(emitted) - len(draft) - 1))
    compact(cache, len(prefix) + consumed, base + (branch_id or 0) * PHRASE, retained_phrases)
    return emitted, accepted, branch_accepted, corrections, pred, len(ids)


def ouroboros_rollout(q, target, prompt, eos, limit, gamma, seed_bank=None):
    sync()
    start = perf_counter()
    bank = CacheEngine(LEVEL, WINDOW)
    if seed_bank is not None:
        bank.token_map = {key: list(phrases) for key, phrases in seed_bank.token_map.items()}
    small = LookaheadDraft(q, bank, prompt)
    target_cache = DynamicCache()
    prefix = list(prompt)
    counts = Counter()
    rounds = []
    while len(prefix) - len(prompt) < limit and prefix[-1] not in eos:
        remaining = limit - len(prefix) + len(prompt)
        before = perf_counter()
        proposal, phrases = small.generate(prefix, min(gamma, remaining), eos, counts)
        sync()
        drafted = perf_counter()
        emitted, accepted, branch, corrections, pred, nodes = verify(
            target, target_cache, prefix, proposal, phrases, eos, remaining)
        sync()
        checked = perf_counter()
        small.cache.crop(min(small.cache.get_seq_length(), len(prefix) + accepted))
        if accepted < len(proposal):
            bank.add_extra_ngram(torch.tensor([proposal[accepted:]]),
                                 torch.tensor([pred[accepted + 1:len(proposal) + 1]]))
        if corrections:
            bank.update_in_place(proposal[-1], corrections)
        prefix.extend(emitted)
        counts.update(draft_ms=1000 * (drafted - before), verify_ms=1000 * (checked - drafted),
                      verify_rounds=1, accept=accepted + branch, accept_main=accepted,
                      accept_phrase=branch, extra_phrase_rounds=int(bool(phrases)),
                      verify_query_tokens=nodes)
        rounds.append(dict(proposal=proposal, candidates=phrases, emitted=emitted,
                           main_accepted=accepted, phrase_accepted=branch))
    sync()
    counts["wall_ms"] = 1000 * (perf_counter() - start)
    return dict(token_ids=prefix[len(prompt):], counts=counts, rounds=rounds)


