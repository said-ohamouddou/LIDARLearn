"""
Base Model Class for Point Cloud Classification

All classification models should inherit from this base class to ensure
a consistent interface across the LidarLearn library.

Author: Said Ohamouddou
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_model_info(config, width: int = 60, mode: str = 'finetune', dataset: str = None) -> None:
    """
    Print a framed message indicating which model is being trained.

    Args:
        config: Configuration object with model parameters. Should have:
            - NAME: The model name (required)
            - base_model: The base/backbone model name (optional, defaults to NAME)
            - finetuning_strategy: The finetuning strategy used (optional)
            - pretrain_dataset: Dataset name for pretraining (optional, auto-detected)
        width: Width of the frame (default: 60)
        mode: 'finetune' or 'pretrain' (default: 'finetune')
        dataset: Dataset name for pretraining (optional, overrides config)

    Example output (finetuning):
        ╔══════════════════════════════════════════════════════════╗
        ║                   Training: PointMAE:IDPT                ║
        ╚══════════════════════════════════════════════════════════╝

    Example output (pretraining):
        ╔══════════════════════════════════════════════════════════╗
        ║              Pretraining: PointBERT on HELIAS            ║
        ╚══════════════════════════════════════════════════════════╝
    """
    # Extract base model and finetuning strategy from config
    base = config.base_model if hasattr(config, 'base_model') else config.NAME
    strategy = config.finetuning_strategy if hasattr(config, 'finetuning_strategy') else ''

    # Build display name
    if strategy:
        display_name = f"{base}:{strategy}"
    else:
        display_name = base

    # Build message based on mode
    if mode == 'pretrain':
        # Get dataset name from parameter or config
        if dataset is None:
            dataset = config.pretrain_dataset if hasattr(config, 'pretrain_dataset') else None
        if dataset:
            message = f"Pretraining: {display_name} on {dataset}"
        else:
            message = f"Pretraining: {display_name}"
    else:
        message = f"Training: {display_name}"

    padding = (width - 2 - len(message)) // 2
    padded_message = " " * padding + message + " " * (width - 2 - padding - len(message))

    print("╔" + "═" * (width - 2) + "╗")
    print("║" + padded_message + "║")
    print("╚" + "═" * (width - 2) + "╝")


class BasePointCloudModel(nn.Module, ABC):
    """
    Abstract base class for all point cloud classification models.

    All models in LidarLearn should inherit from this class and implement
    the required abstract methods to ensure a unified API.

    Required Methods:
        - forward(x): Forward pass through the model
        - get_loss_acc(pred, gt, smoothing): Calculate loss and accuracy

    Example usage::

        class MyModel(BasePointCloudModel):
            def __init__(self, config, **kwargs):
                super().__init__(config)
                # Initialize your model layers here

            def forward(self, x):
                # Implement forward pass
                return logits

            def get_loss_acc(self, pred, gt, smoothing=None):
                # Implement loss and accuracy calculation
                return loss, acc
    """

    def __init__(self, config, num_classes=40, **kwargs):
        """
        Initialize the base model.

        Args:
            config: Configuration object with model parameters
            num_classes (int): Number of output classes (default: 40)
            **kwargs: Additional keyword arguments
        """
        super().__init__()

        # Extract common parameters from config
        self.num_classes = config.num_classes if hasattr(config, 'num_classes') else num_classes
        self.label_smoothing = config.label_smoothing if hasattr(config, 'label_smoothing') else 0.0

        # Store config for reference
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        This method MUST be implemented by all subclasses.

        Args:
            x (torch.Tensor): Input point cloud data
                - Expected shape: [B, N, C] or [B, C, N]
                - B: Batch size
                - N: Number of points
                - C: Number of channels (typically 3 for xyz, or 6 for xyz+normals)

        Returns:
            torch.Tensor: Classification logits of shape [B, num_classes]

        Note:
            - Models should handle both input formats [B, N, C] and [B, C, N]
            - Output should always be [B, num_classes] logits (before softmax)
        """
        pass

    def get_loss_acc(self, pred: torch.Tensor, gt: torch.Tensor,
                     smoothing: float = None) -> tuple:
        """
        Calculate loss and accuracy.

        Args:
            pred (torch.Tensor): Model predictions/logits of shape [B, num_classes]
            gt (torch.Tensor): Ground truth labels of shape [B]
            smoothing (float, optional): Label smoothing value.
                                        If None, uses self.label_smoothing

        Returns:
            tuple: (loss, accuracy)
                - loss (torch.Tensor): Scalar loss value
                - accuracy (torch.Tensor): Accuracy as percentage (0-100)
        """
        loss = self._compute_loss_with_smoothing(pred, gt, smoothing)
        acc = self._compute_accuracy(pred, gt)
        return loss, acc

    def _compute_loss_with_smoothing(self, pred: torch.Tensor, gt: torch.Tensor,
                                     smoothing: float = None) -> torch.Tensor:
        """
        Helper method to compute cross-entropy loss with optional label smoothing.

        This is a convenience method that subclasses can use in their
        get_loss_acc implementation.

        Args:
            pred (torch.Tensor): Predictions of shape [B, num_classes]
            gt (torch.Tensor): Ground truth labels of shape [B]
            smoothing (float, optional): Label smoothing value

        Returns:
            torch.Tensor: Scalar loss value
        """
        gt = gt.contiguous().view(-1).long()

        if smoothing is None:
            smoothing = self.label_smoothing

        if smoothing > 0:
            eps = smoothing
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt)

        return loss

    def _compute_accuracy(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Helper method to compute classification accuracy.

        This is a convenience method that subclasses can use in their
        get_loss_acc implementation.

        Args:
            pred (torch.Tensor): Predictions of shape [B, num_classes]
            gt (torch.Tensor): Ground truth labels of shape [B]

        Returns:
            torch.Tensor: Accuracy as percentage (0-100)
        """
        gt = gt.contiguous().view(-1).long()
        pred_cls = pred.argmax(-1)
        acc = (pred_cls == gt).sum() / float(gt.size(0))
        return acc * 100

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Get the number of parameters in the model.

        Args:
            trainable_only (bool): If True, count only trainable parameters.
                                  If False, count all parameters.

        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"{self.__class__.__name__}("
                f"num_classes={self.num_classes}, "
                f"label_smoothing={self.label_smoothing}, "
                f"params={self.get_num_parameters():,})")
