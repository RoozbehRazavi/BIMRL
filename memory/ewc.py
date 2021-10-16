from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def variable(t: torch.Tensor, **kwargs):
    t = t.to(device)
    return Variable(t)


class EWC(object):
    def __init__(self, model: nn.Module, dataloader: list):

        self.model = model
        self.dataloader = dataloader

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for brim_hidden, state in self.dataloader:
            self.model.zero_grad()
            brim_hidden = variable(brim_hidden)
            state = variable(state)
            output = self.model(brim_hidden, state)
            loss = self.model.loss(brim_hidden, output[0], output[1], output[2])
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

    @staticmethod
    def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
                  ewc, importance: float):
        model.train()
        epoch_loss = 0
        for brim_hidden, state in data_loader:
            if brim_hidden.dim() == 3:
                brim_hidden = brim_hidden.view(-1, brim_hidden.shape[-1])
            if state.dim() == 3:
                state = state.view(-1, state.shape[-1])

            brim_hidden, state = variable(brim_hidden), variable(state)
            optimizer.zero_grad()
            output = model(brim_hidden, state)
            loss = model.loss(brim_hidden, output[0], output[1], output[2]) + importance * ewc.penalty(model)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        return epoch_loss / len(data_loader)

    @staticmethod
    def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
        epoch_loss = 0
        model.train()
        for brim_hidden, state in data_loader:
            if brim_hidden.dim() == 3:
                brim_hidden = brim_hidden.view(-1, brim_hidden.shape[-1])
            if state.dim() == 3:
                state = state.view(-1, state.shape[-1])
            brim_hidden = variable(brim_hidden)
            state = variable(state)
            optimizer.zero_grad()
            output = model(brim_hidden, state)
            loss = model.loss(brim_hidden, output[0], output[1], output[2])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        return epoch_loss / len(data_loader)


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct / len(data_loader.dataset)
