
'''
Giving an N x M attention matrix, returns the same matrix,
but performs masking to determine where to block gradients.
'''

import numpy
import torch
from torch.autograd import Variable

from BRIM.sparse_attn import Sparse_attention


class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0

class Sparse_grad_attention(torch.autograd.Function):
    # def __init__(self, top_k):
    #     super(Sparse_grad_attention,self).__init__()
    #
    #     self.sa = Sparse_attention(top_k=top_k)

    @staticmethod
    def forward(ctx, inp, sa):
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)

        return inp

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        return (grad_output) * (sparsified > 0.0).float()

