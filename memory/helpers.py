import torch
import torch.nn.functional as F


def compute_RPE(rew_raw, prev_state, next_state, action, brim_output5, rew_reconstruction):
    if next_state is None:
        return torch.ones_like(rew_raw)
    _, reward_perd = rew_reconstruction(brim_output5, prev_state, next_state, action, rew_raw, return_predictions=True)
    return rew_raw - reward_perd


def compute_weight(key1, key2, key3, query):
    key = list()

    if key1 is not None:
        key.append(key1.unsqueeze(0))

    if key2 is not None:
        key.append(key2.unsqueeze(0))

    if key3 is not None:
        key.append(key3.unsqueeze(0))

    key = torch.cat(key, dim=0)
    key = key.permute(1, 0, 2)
    query = query.permute(0, 2, 1)
    A = torch.bmm(key, query)
    weight = spatial_softmax(A)
    return weight


def spatial_softmax(A):
    A = F.softmax(A, dim=-1)
    return A


def apply_alpha(A, V):
    A = A.permute(0, 2, 1)
    V = V.permute(1, 0, 2)
    result = torch.bmm(A, V)
    return result
