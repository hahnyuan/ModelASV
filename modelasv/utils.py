import torch
import torch.nn.functional as F


def fake_linear(input, weight, bias=None):
    shape = [_ for _ in input.shape]
    shape[-1] = weight.shape[0]
    return torch.zeros(shape, dtype=input.dtype, device=input.device)


def do_fake_inference():
    # not compute anything, just return a zero tensor with the expected output
    F.linear = fake_linear
