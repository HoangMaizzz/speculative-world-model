"""Adapt the pinned SafeAILab EAGLE-3 inference code to the SpecJAX Qwen head."""

from time import perf_counter

EAGLE_REVISION = "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"
HEAD_MODEL = "thoughtworks/Qwen2.5-7B-Instruct-Eagle3"
HEAD_REVISION = "ff17dda64a036cf5bd7bc56c0ab728325f1c0d0b"
AUX_LAYERS = (1, 13, 24)  # Zero-based decoder layers, captured after each layer.


def check_vocab_map(offsets, target_mask, draft_size, target_size):
    """The SpecJAX export stores signed offsets from draft IDs in d2t."""
    if len(offsets) != draft_size or len(target_mask) != target_size:
        raise ValueError("Unexpected EAGLE-3 vocabulary mapping shape")
    mapped = [i + value for i, value in enumerate(offsets)]
    if mapped != [i for i, keep in enumerate(target_mask) if keep]:
        raise ValueError("EAGLE-3 d2t offsets and t2d mask disagree")


def trace_parents(paths, depths):
    """Recover actual node parents from the upstream root-to-leaf retrieval table."""
    parents = {0: -1}
    if not depths or depths[0] != 0:
        raise ValueError("Invalid EAGLE tree root")
    for path in paths:
        if not path or path[0] != 0:
            raise ValueError("EAGLE path does not start at the verifier root")
        for depth, (parent, child) in enumerate(zip(path, path[1:]), 1):
            if not 0 <= parent < child < len(depths) or depths[child] != depth:
                raise ValueError("EAGLE path violates node order or depth")
            if parents.setdefault(child, parent) != parent:
                raise ValueError("EAGLE node has multiple parents")
    if len(parents) != len(depths):
        raise ValueError("Missing EAGLE tree nodes in the retrieval table")
    return [parents[node] for node in range(len(depths))]


class EagleQwen:
    """Expose the interface consumed by the author's initialize/verify/update utilities."""

    use_eagle3 = True

    def __init__(self, base_model, ea_layer, aux_layers, cache_length):
        from eagle.model.kv_cache import initialize_past_key_values

        self.base_model = base_model
        self.ea_layer = ea_layer
        self.aux_indices = tuple(i + 1 for i in aux_layers)  # HF tuple starts at embedding.
        self.cache, self.cache_data, self.cache_lengths = initialize_past_key_values(
            base_model, max_length=cache_length)
        self.input_device = base_model.model.embed_tokens.weight.device
        self.head_device = ea_layer.lm_head.weight.device

    def __call__(self, input_ids, *, past_key_values, output_orig, position_ids=None):
        outputs = self.base_model.model(
            input_ids=input_ids, past_key_values=past_key_values, position_ids=position_ids,
            output_hidden_states=True, use_cache=True, return_dict=True)
        features = tuple(outputs.hidden_states[i].to(self.head_device) for i in self.aux_indices)
        hidden = outputs.last_hidden_state if position_ids is not None else outputs.last_hidden_state[:, -1:]
        logits = self.base_model.lm_head(hidden)
        return {"hidden_states": features}, logits, outputs.last_hidden_state

    def reset(self):
        from eagle.model.utils import reset_tree_mode

        self.cache_lengths.zero_()
        reset_tree_mode(self)
        self.ea_layer.reset_kv()

    def decode_reference(self, prompt, eos, limit):
        """Independent greedy token-at-a-time decode with the same Qwen weights and cache."""
        import torch
        from suffix_lab.runtime import sync_model

        self.reset()
        start = perf_counter()
        input_ids = torch.tensor([prompt], device=self.input_device)
        generated = []
        for _ in range(limit):
            output = self.base_model.model(input_ids, past_key_values=self.cache,
                                           use_cache=True, return_dict=True)
            token = int(self.base_model.lm_head(output.last_hidden_state[:, -1]).argmax())
            generated.append(token)
            if token in eos:
                break
            input_ids = torch.tensor([[token]], device=self.input_device)
        sync_model(self.base_model)
        return {"token_ids": generated, "wall_ms": 1000 * (perf_counter() - start)}

    def decode_tree(self, prompt, eos, limit):
        import torch
        from eagle.model.utils import (evaluate_posterior, initialize_tree, tree_decoding,
                                       update_inference_inputs)
        from suffix_lab.long60_plan import until_eos
        from suffix_lab.runtime import sync_model

        self.reset()
        start = perf_counter()
        prefix = torch.tensor([prompt], device=self.input_device)
        padding = prefix.new_full((1, 1), -1)
        draft, retrieve, mask, positions, _, _, seed = initialize_tree(prefix, self, self.cache, None)
        sync_model(self.base_model)
        first_tree_ms = 1000 * (perf_counter() - start)
        generated, rounds = [], []
        while len(generated) < limit:
            tick = perf_counter()
            self.base_model.model.tree_mask = mask.to(self.input_device)
            draft = draft.to(self.input_device)
            logits, features, _ = tree_decoding(
                self, draft, self.cache, positions.to(self.input_device), prefix, retrieve)
            candidates = torch.cat((draft, padding), dim=1)[0, retrieve]
            best, accepted, next_logits = evaluate_posterior(logits, candidates, None)
            next_seed = int(next_logits.argmax())
            best_index, accepted_count = int(best), int(accepted)
            sync_model(self.base_model)
            verify_ms = 1000 * (perf_counter() - tick)

            tokens = draft[0].tolist()
            depths = positions.tolist()
            paths = [[node for node in row if node >= 0] for row in retrieve.tolist()]
            parents = trace_parents(paths, depths)
            if len(tokens) != len(depths) or tokens[0] != int(seed):
                raise AssertionError("Verifier root or node trace changed")
            accepted_nodes = paths[best_index][:accepted_count + 1]
            native_emitted = [tokens[node] for node in accepted_nodes]
            committed = until_eos(native_emitted, eos)[:limit - len(generated)]
            rounds.append(dict(offset=len(generated), tokens=tokens, depths=depths, parents=parents,
                               paths=paths, best_path=best_index,
                               committed_nodes=accepted_nodes[:len(committed)], committed=committed,
                               accepted_draft=len(committed) - 1, next_seed=next_seed,
                               verify_ms=verify_ms, update_and_next_draft_ms=None))
            generated.extend(committed)
            if len(generated) == limit or generated[-1] in eos:
                break

            tick = perf_counter()
            prefix, draft, retrieve, mask, positions, _, _, seed = update_inference_inputs(
                prefix, candidates, best, accepted, retrieve, None, 0, self.cache_data,
                self.cache_lengths, self, features, next_logits)
            sync_model(self.base_model)
            rounds[-1]["update_and_next_draft_ms"] = 1000 * (perf_counter() - tick)
            if prefix.shape[1] != len(prompt) + len(generated) or int(seed) != next_seed:
                raise AssertionError("EAGLE cache update changed the committed prefix or next root")
            if not bool((self.cache_lengths == prefix.shape[1]).all()):
                raise AssertionError("Target cache was not compacted to the accepted path")
            if self.ea_layer.stable_kv[0][0].shape[2] != prefix.shape[1]:
                raise AssertionError("Drafter cache no longer represents the accepted prefix")
        return dict(token_ids=generated, rounds=rounds, first_tree_ms=first_tree_ms,
                    wall_ms=1000 * (perf_counter() - start))


def load_model(target_path, head_path, cache_length, nodes, depth, top_k):
    import torch
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    from eagle.model.modeling_qwen2_kv import Qwen2ForCausalLM
    from safetensors.torch import load_file

    mapping = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    mapping.update({f"model.layers.{i}": int(i >= 18) for i in range(28)})
    target = Qwen2ForCausalLM.from_pretrained(
        str(target_path), local_files_only=True, use_safetensors=True, device_map=mapping,
        torch_dtype=torch.float16, attn_implementation="eager").eval()
    cfg = EConfig.from_pretrained(str(head_path / "config.json"))
    if (target.config.num_hidden_layers, cfg.num_hidden_layers, cfg.num_key_value_heads,
            cfg.draft_vocab_size, cfg.hidden_size, cfg.vocab_size) != (28, 1, 7, 32000, 3584, 152064):
        raise ValueError("Unexpected architecture for the pinned Qwen EAGLE-3 pair")
    cfg.max_position_embeddings = cache_length
    head = Model(cfg, total_tokens=nodes, depth=depth, top_k=top_k, load_emb=False)
    weights = load_file(head_path / "model.safetensors")
    if (weights["d2t"].ndim != 1 or weights["t2d"].ndim != 1
            or weights["d2t"].dtype not in (torch.int32, torch.int64)
            or weights["t2d"].dtype != torch.bool):
        raise ValueError("Unexpected EAGLE-3 vocabulary mapping tensors")
    check_vocab_map(weights["d2t"].tolist(), weights["t2d"].tolist(),
                    cfg.draft_vocab_size, cfg.vocab_size)

    # SpecJAX borrows the frozen target embedding. Its checkpoint omits that tensor.
    embedding = target.model.embed_tokens.weight.detach().to("cuda:1")
    if "embed_tokens.weight" in weights and not torch.equal(
            weights["embed_tokens.weight"].to(device=embedding.device, dtype=embedding.dtype), embedding):
        raise ValueError("Draft embedding disagrees with the pinned Qwen target")
    head.embed_tokens.weight = torch.nn.Parameter(embedding, requires_grad=False)
    weights["embed_tokens.weight"] = embedding
    head.load_state_dict(weights, strict=True)
    head = head.to(device="cuda:1", dtype=torch.float16).eval()
    head.init_tree()
    return EagleQwen(target, head, AUX_LAYERS, cache_length)
