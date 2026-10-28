"""Sparse decoder outputs before final norm; hooks never modify model tensors."""
class HiddenCapture:
    def __init__(self, model, layers, positions, final_norm=False):
        self.model, self.layers, self.positions = model, list(layers), list(positions)
        self.final_norm = final_norm
        self.handles, self.values = [], {}

    def __enter__(self):
        import torch
        def hook(key):
            def capture(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                idx = torch.tensor(self.positions, device=value.device)
                self.values[key] = value[0].index_select(0, idx).detach().to(dtype=torch.float16, device="cpu").contiguous()
            return capture
        for i in self.layers:
            self.handles.append(self.model.model.layers[i].register_forward_hook(hook(f"layer_{i}")))
        if self.final_norm:
            self.handles.append(self.model.model.norm.register_forward_hook(hook("final_norm")))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
