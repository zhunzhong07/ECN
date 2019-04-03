import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math


class ExemplarMemory(Function):
    def __init__(self, em, alpha=0.01):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.alpha = alpha

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)
        for x, y in zip(inputs, targets):
            self.em[y] = self.alpha * self.em[y] + (1. - self.alpha) * x
            self.em[y] /= self.em[y].norm()
        return grad_inputs, None


# Invariance learning loss
class InvNet(nn.Module):
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01):
        super(InvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature fact
        self.knn = knn  # Knn for neighborhood invariance

        # Exemplar memory
        self.em = nn.Parameter(torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets, epoch=None):

        alpha = self.alpha * epoch
        inputs = ExemplarMemory(self.em, alpha=alpha)(inputs, targets)

        inputs /= self.beta
        if self.knn > 0 and epoch > 4:
            # With neighborhood invariance
            loss = self.smooth_loss(inputs, targets)
        else:
            # Without neighborhood invariance
            loss = F.cross_entropy(inputs, targets)
        return loss

    def smooth_loss(self, inputs, targets):
        targets = self.smooth_hot(inputs.detach().clone(), targets.detach().clone(), self.knn)
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, inputs, targets, k=6):
        # Sort
        _, index_sorted = torch.sort(inputs, dim=1, descending=True)

        ones_mat = torch.ones(targets.size(0), k).to(self.device)
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)

        weights = F.softmax(ones_mat, dim=1)
        targets_onehot.scatter_(1, index_sorted[:, 0:k], ones_mat * weights)
        targets_onehot.scatter_(1, targets, float(1))

        return targets_onehot



