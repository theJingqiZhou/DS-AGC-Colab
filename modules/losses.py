# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import torch
from torch import autograd, nn
from torch.nn import functional as F


def diff_loss(
    diff: torch.Tensor, s: torch.Tensor, falpha: torch.Tensor
) -> torch.Tensor:
    """
    compute the 1st loss of L_{graph_learning}
    """
    if len(s.shape) == 4:
        # batch input
        return falpha * torch.mean(torch.sum(torch.sum(diff**2, dim=3) * s, dim=(1, 2)))
    return falpha * torch.sum(torch.matmul(s, torch.sum(diff**2, dim=2)))


def f_norm_loss(s: torch.Tensor, falpha: float) -> torch.Tensor:
    """
    compute the 2nd loss of L_{graph_learning}
    """
    if len(s.shape) == 3:
        # batch input
        return falpha * torch.sum(torch.mean(s**2, dim=0))
    else:
        return falpha * torch.sum(s**2)


def get_cos_similarity_distance(
    pseudo: torch.Tensor, pred: torch.Tensor
) -> torch.Tensor:
    """Get distance in cosine similarity
    :param features: features of samples, (batch_size, num_clusters)
    :return: distance matrix between features, (batch_size, batch_size)
    """
    pseudo_norm = torch.norm(pseudo, dim=1, keepdim=True)
    pseudo = pseudo / pseudo_norm

    pred_norm = torch.norm(pred, dim=1, keepdim=True)
    pred = pred / pred_norm

    cos_dist_matrix = torch.mm(pseudo, pred.transpose(0, 1))
    return cos_dist_matrix


class ReverseLayerF(autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReverseF(autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input_tensor: torch.Tensor, coeff: float | None = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input_tensor * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

    The forward and backward behaviours are:

    .. math::
        \mathcal{R}(x) = x,

        \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

    :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

    .. math::
        \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

    where :math:`i` is the iteration step.

    Parameters:
        - **alpha** (float, optional): :math:`α`. Default: 1.0
        - **lo** (float, optional): Initial value of :math:`\lambda`. Default: 0.0
        - **hi** (float, optional): Final value of :math:`\lambda`. Default: 1.0
        - **max_iters** (int, optional): :math:`N`. Default: 1000
        - **auto_step** (bool, optional): If True, increase :math:`i` each time `forward` is called.
          Otherwise use function `step` to increase :math:`i`. Default: False
    """

    def __init__(
        self,
        alpha: float = 1.0,
        lo: float = 0.0,
        hi: float = 1.0,
        max_iters: int = 1000,
        auto_step: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, x: torch.Tensor) -> Any:
        """"""
        coeff = float(
            2.0
            * (self.hi - self.lo)
            / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo)
            + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseF.apply(x, coeff)

    def step(self) -> None:
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100.0 / batch_size)
        return correct


class Discriminator3(nn.Module):
    def __init__(self, hidden_1: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        # self.fc2 = nn.Linear(hidden_1, 1)
        self.fc2 = nn.Linear(hidden_1, 3)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_1: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        # self.fc2=nn.Linear(hidden_1, 1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class TripleDomainAdversarialLoss(nn.Module):
    def __init__(self, hidden_1: int, max_iter: int = 100) -> None:
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=1.0, max_iters=max_iter, auto_step=True
        )
        self.domain_discriminator = Discriminator3(hidden_1)
        self.domain_discriminator_accuracy = None
        self.crition = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.grl(x)
        d = self.domain_discriminator(f)
        source_num = int(len(x) / 3)
        d_label_s = torch.ones((source_num, 1)).to(x.device)
        d_label_t = torch.zeros((source_num, 1)).to(x.device)
        d_label_u = torch.ones((source_num, 1)).to(x.device) + 1
        label = torch.cat((d_label_s, d_label_t, d_label_u)).squeeze().long()
        # self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return self.crition(d, label)


class DomainAdversarialLoss(nn.Module):
    def __init__(
        self, hidden_1: int, reduction: str = "mean", max_iter: int = 100
    ) -> None:
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=1.0, max_iters=max_iter, auto_step=True
        )
        self.domain_discriminator = Discriminator(hidden_1)
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.grl(x)
        d = self.domain_discriminator(f)

        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones(len(d_s), 1).to(x.device)
        d_label_t = torch.zeros(len(d_t), 1).to(x.device)

        self.domain_discriminator_accuracy = 0.5 * (
            binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t)
        )
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class DomainAdversarialLossThreeada(nn.Module):
    def __init__(
        self, hidden_1: int, reduction: str = "mean", max_iter: int = 100
    ) -> None:
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=1.0, max_iters=max_iter, auto_step=True
        )
        self.domain_discriminator = Discriminator(hidden_1)
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.grl(x)
        d = self.domain_discriminator(f)

        source_num = int(len(x) / 3)
        d_s = d[0 : 2 * source_num, :]
        d_t = d[2 * source_num :, :]
        d_label_s = torch.ones(2 * source_num, 1).to(x.device)
        d_label_t = torch.zeros(source_num, 1).to(x.device)

        self.domain_discriminator_accuracy = 0.5 * (
            binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t)
        )
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
    ```
        loss1 = 1
        loss2 = 2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    ```
    """

    def __init__(self, num: int = 2) -> None:
        super().__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)
        print(self.params)

    def forward(self, *x: tuple[torch.Tensor, ...]) -> torch.Tensor:
        assert isinstance(x[0], torch.Tensor)
        loss_sum = torch.tensor(0).to(x[0])
        # length = len(x) - 1
        for i, loss in enumerate(x):
            assert isinstance(loss, torch.Tensor)
            loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            # if i == length:
            #     loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            # else:
            #     loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
        return loss_sum


if __name__ == "__main__":
    awl = AutomaticWeightedLoss(4)
    awl(2.5, 2.6, 3.7, 3.8)
    print(awl.parameters())
