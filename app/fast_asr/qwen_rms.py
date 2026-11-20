"""Opt-in eager Qwen2 RMSNorm for an isolated inference process running on Ascend."""
import copy

import torch


def _output(op, hidden, weight, epsilon):
    return op(hidden, weight, epsilon)[0]


@torch.inference_mode()
def _check_operation(norm, op, length):
    # Include low-energy inputs so an incorrect epsilon or normalization axis is observable.
    weight = norm.weight
    hidden_size = weight.numel()
    pattern = torch.arange(2*length*hidden_size, dtype=torch.float32)
    pattern = ((pattern % 313) - 114).reshape(2,length,hidden_size) / 128
    pattern[1] *= 0.0003
    pattern[...,::31] = 0
    x = pattern.to(device=weight.device,dtype=weight.dtype)
    expected = norm(x)
    actual = _output(op,x,weight,norm.variance_epsilon)
    if actual.shape != x.shape or actual.dtype != expected.dtype or actual.device != x.device:
        raise ValueError("npu_rms_norm output metadata differs from Qwen2RMSNorm")
    a,b = actual.detach().float().cpu(),expected.detach().float().cpu()
    rtol = {torch.bfloat16:0.017,torch.float16:0.0022,torch.float32:3e-5}[weight.dtype]
    if not torch.allclose(a,b,atol=2e-5,rtol=rtol):
        raise ValueError("device npu_rms_norm failed the original layer's numerical probe")
    return {"sequence_length":length,"values":a.numel(),"exact_values":int((a == b).sum()),
            "max_abs_error":float((a-b).abs().max()),"rtol":rtol,"atol":2e-5}


class Installed:
    def __init__(self,norms,op,probes):
        self._hooks = []
        self._calls = 0
        self._total_layers = len(norms)
        self._probes = probes
        self._device = str(norms[0].weight.device)
        self._dtype = str(norms[0].weight.dtype)
        self._epsilon = float(norms[0].variance_epsilon)
        try:
            for mod in norms:
                def forward(hidden_states, _layer=mod):
                    out = _output(op,hidden_states,_layer.weight,_layer.variance_epsilon)
                    self._calls += 1
                    return out
                mod.forward = forward
                self._hooks.append((mod,forward))
        except BaseException:
            self.uninstall()
            raise

    def snapshot(self):
        return {"enabled":bool(self._hooks),"device":self._device,"dtype":self._dtype,
                "epsilon":self._epsilon,"layers":self._total_layers,"calls":self._calls,
                "probe":copy.deepcopy(self._probes)}

    def uninstall(self):
        # Deleting the instance method restores the class's original descriptor.
        for mod,forward in reversed(self._hooks):
            if vars(mod).get("forward") is forward:
                del mod.forward
        self._hooks.clear()


def install(model, *, op=None, require_npu=True):
    """Patch only this loaded eval model while idle. Probes run before changing any layer."""
    cfg = model.config
    if model.training or getattr(cfg, "model_type", None) != "qwen2":
        raise ValueError("expected an eval-mode Qwen2 model")
    norms = [m for m in model.modules() if type(m).__name__ == "Qwen2RMSNorm"]
    if len(norms) != 2*cfg.num_hidden_layers+1:
        raise ValueError("unexpected number of Qwen2RMSNorm modules")
    first = norms[0].weight
    if require_npu and first.device.type != "npu":
        raise ValueError("expected Ascend weights")
    if first.dtype not in (torch.bfloat16,torch.float16,torch.float32):
        raise ValueError("unsupported normalization dtype")
    if any("forward" in vars(m) or m.training or m.weight.shape != (cfg.hidden_size,)
           or m.weight.device != first.device or m.weight.dtype != first.dtype
           or m.variance_epsilon != cfg.rms_norm_eps for m in norms):
        raise ValueError("unexpected norm settings or an existing instance-level override")
    if op is None:
        import torch_npu
        op = torch_npu.npu_rms_norm
    probes = []
    for i in sorted({0,len(norms)//2,len(norms)-1}):
        for length in (1,7):
            probes.append({"layer_index":i, **_check_operation(norms[i],op,length)})
    return Installed(norms,op,probes)
