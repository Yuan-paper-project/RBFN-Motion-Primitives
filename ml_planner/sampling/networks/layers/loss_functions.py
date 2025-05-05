__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from torch import nn
from torch.nn import functional as F


class PositionLoss(nn.MSELoss):
    """Loss function for the position prediction"""
    def __init__(self, labels, size_average=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduction=reduction)
        self.labels = labels
        self.idxs = [self.labels.index(i) for i in ['x', 'y']]

    def name(self) -> str:
        return self.__class__.__name__

    def forward(self, pred, gt):
        gt_pos = gt[..., self.idxs]
        pred_pos = pred[..., self.idxs]
        loss = F.mse_loss(pred_pos, gt_pos, reduction=self.reduction)
        if self.reduction == "sum":
            loss = loss / pred.shape[0]
        return loss


class OrientationLoss(nn.MSELoss):
    """Loss function for the orientation prediction"""
    def __init__(self, labels, size_average=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduction=reduction)
        self.labels = labels
        self.idxs = self.labels.index("psi")

    def name(self) -> str:
        return self.__class__.__name__

    def forward(self, pred, gt):
        gt_psi = gt[..., self.idxs]
        pred_psi = pred[..., self.idxs]
        loss = F.mse_loss(pred_psi, gt_psi, reduction=self.reduction)
        if self.reduction == "sum":
            loss = loss / pred.shape[0]
        return loss*100


class VelocityLoss(nn.MSELoss):
    """Loss function for the velocity prediction"""
    def __init__(self, labels, size_average=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduction=reduction)
        self.labels = labels
        self.idxs = self.labels.index("v")

    def name(self) -> str:
        return self.__class__.__name__

    def forward(self, pred, gt):
        gt_v = gt[..., self.idxs]
        pred_v = pred[..., self.idxs]
        loss = F.mse_loss(pred_v, gt_v, reduction=self.reduction)
        if self.reduction == "sum":
            loss = loss / pred.shape[0]
        return loss


class AccelerationLoss(nn.MSELoss):
    """Loss function for the acceleration prediction"""
    def __init__(self, labels, size_average=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduction=reduction)
        self.labels = labels
        self.idxs = self.labels.index("acc")

    def name(self) -> str:
        return self.__class__.__name__

    def forward(self, pred, gt):
        gt_a = gt[..., self.idxs]
        pred_a = pred[..., self.idxs]
        loss = F.mse_loss(pred_a, gt_a, reduction=self.reduction)
        if self.reduction == "sum":
            loss = loss / pred.shape[0]
        return loss


class SteeringLoss(nn.MSELoss):
    """Loss function for the steering angle"""
    def __init__(self, labels, size_average=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduction=reduction)
        self.labels = labels
        self.idxs = self.labels.index("delta")

    def name(self) -> str:
        return self.__class__.__name__

    def forward(self, pred, gt):
        gt_steering = gt[..., self.idxs]
        pred_steering = pred[..., self.idxs]
        loss = F.mse_loss(pred_steering, gt_steering, reduction=self.reduction)
        if self.reduction == "sum":
            loss = loss / pred.shape[0]
        return loss
