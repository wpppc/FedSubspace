# subspace/utils.py
import torch

def flatten_lora_params(lora_state_dict):
    """
    将 LoRA 参数按照 key 排序并展开为一个长向量。
    返回：
        vec: (D,) tensor
        meta: 每个层的 (key, shape, numel)
    """
    keys = sorted(lora_state_dict.keys())
    vecs = []
    meta = []
    for k in keys:
        t = lora_state_dict[k].reshape(-1).float()
        vecs.append(t)
        meta.append((k, list(lora_state_dict[k].shape), t.numel()))
    return torch.cat(vecs, dim=0), meta


def unflatten_lora_params(vec, meta):
    """
    根据 meta 中的结构信息，将扁平化 vec 重构为 state_dict。
    """
    state = {}
    offset = 0
    for (k, shape, numel) in meta:
        part = vec[offset: offset + numel]
        state[k] = part.reshape(shape)
        offset += numel
    return state
