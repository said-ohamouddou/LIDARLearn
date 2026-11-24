"""
Validation Module
Handles validation logic during training
"""

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    matthews_corrcoef,
    accuracy_score
)
from sklearn.svm import LinearSVC

from utils import misc, dist_utils
from utils.logger import print_log
from .metrics_tracker import Acc_Metric


def validate(base_model, test_dataloader, epoch, args, config,
             metrics_tracker, logger=None):
    """Validate the model and update metrics tracker."""

    base_model.eval()  # set model to eval mode
    test_pred = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            logits = base_model(points)

            # Handle tuple/list returns
            if isinstance(logits, (tuple, list)) and len(logits) > 1:
                logits = logits[0]

            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)
            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        # Convert to numpy for sklearn metrics
        test_pred_np = test_pred.cpu().numpy()
        test_label_np = test_label.cpu().numpy()

        # Calculate current epoch metrics using sklearn
        acc = accuracy_score(test_label_np, test_pred_np) * 100.
        balanced_acc = balanced_accuracy_score(test_label_np, test_pred_np) * 100.

        # Calculate current F1 scores
        f1_macro = f1_score(
            test_label_np,
            test_pred_np,
            average='macro',
            zero_division=0
        ) * 100.
        f1_weighted = f1_score(
            test_label_np,
            test_pred_np,
            average='weighted',
            zero_division=0
        ) * 100.

        # Calculate current precision
        precision_macro = precision_score(
            test_label_np,
            test_pred_np,
            average='macro',
            zero_division=0
        ) * 100.
        precision_weighted = precision_score(
            test_label_np,
            test_pred_np,
            average='weighted',
            zero_division=0
        ) * 100.


        # Calculate current kappa and MCC
        kappa = cohen_kappa_score(test_label_np, test_pred_np) * 100.
        mcc = matthews_corrcoef(test_label_np, test_pred_np) * 100.

        # Update metrics tracker with current results
        is_best = metrics_tracker.update(
            epoch,
            acc,
            balanced_acc,
            test_pred_np,
            test_label_np
        )

        print_log(
            '[Validation] EPOCH: %d  acc = %.4f%%, balanced_acc = %.4f%%, f1_macro = %.4f%%, kappa = %.4f%%' %
            (epoch, acc, balanced_acc, f1_macro, kappa),
            logger=logger
        )

        # Print best metrics summary
        metrics_tracker.print_best_metrics(logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Return metrics along with current epoch values for history tracking
    return Acc_Metric(acc), {
        'val_acc': acc,
        'val_balanced_acc': balanced_acc,
        'val_f1_macro': f1_macro,
        'val_f1_weighted': f1_weighted,
        'val_precision_macro': precision_macro,
        'val_precision_weighted': precision_weighted,
        'val_kappa': kappa,
        'val_mcc': mcc,
    }


def validate_svm(base_model, train_dataloader, test_dataloader,
                 epoch, args, config, *,
                 npoints, svm_kwargs=None, return_percent=True,
                 val_writer=None, writer_tag='Metric/SVM_ACC',
                 logger=None):
    """SVM linear-probe validation for self-supervised pretraining.

    Extracts features from the frozen encoder via ``base_model(points, noaug=True)``,
    fits a LinearSVC on the train split, and reports accuracy on the test split.

    Args:
        base_model: The model (wrapped in DataParallel/DDP). Must support a
            ``noaug=True`` keyword that returns a feature vector per sample.
        train_dataloader: Dataloader whose batches are either
            ``(taxonomy_id, model_id, (points, label))`` or
            ``(taxonomy_id, model_id, points, img, text, label)`` (ReCon).
        test_dataloader: Same format as train_dataloader.
        epoch (int): Current epoch number (used for writer logging).
        args: Experiment args (needs ``args.distributed``).
        config: Experiment config.
        npoints (int): Number of points — features are FPS-sampled to this count
            if the batch doesn't already have the right size.
        svm_kwargs (dict, optional): Extra keyword arguments forwarded to
            ``LinearSVC`` (e.g. ``{'C': 0.075}`` for ReCon). Defaults to ``{}``.
        return_percent (bool): If True the returned ``Acc_Metric`` stores the
            accuracy as a percentage (e.g. 83.5). If False it stores a fraction
            (0.835). Defaults to True.
        val_writer: TensorBoard SummaryWriter or None.
        writer_tag (str): Tag name used when writing to val_writer.
        logger: Logger instance or name string.

    Returns:
        Acc_Metric: SVM accuracy (percentage if ``return_percent=True``).
    """
    if svm_kwargs is None:
        svm_kwargs = {}

    base_model.eval()
    train_features, train_labels = [], []
    test_features, test_labels = [], []

    def _extract(dataloader, feature_list, label_list):
        for batch in dataloader:
            # Two supported batch formats:
            #   (taxonomy_id, model_id, (points, label))   — standard
            #   (taxonomy_id, model_id, points, img, text, label) — ReCon
            if len(batch) == 3:
                _, _, data = batch
                if not isinstance(data, (list, tuple)):
                    return False          # unexpected format — abort
                points, label = data[0].cuda(), data[1].cuda()
            else:
                # ReCon: (taxonomy_id, model_id, pc, img, text, label)
                points, label = batch[2].cuda(), batch[5].cuda()

            if points.size(1) != npoints:
                points = misc.fps(points, npoints)

            feature = base_model(points, noaug=True)
            if isinstance(feature, tuple) or feature.dim() == 0:
                return False              # model returned scalar or tuple — abort
            feature_list.append(feature.detach())
            label_list.append(label.view(-1).detach())
        return True

    with torch.no_grad():
        if not _extract(train_dataloader, train_features, train_labels):
            return Acc_Metric(0.)
        if not _extract(test_dataloader, test_features, test_labels):
            return Acc_Metric(0.)

        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_labels = dist_utils.gather_tensor(train_labels, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_labels = dist_utils.gather_tensor(test_labels, args)

        clf = LinearSVC(**svm_kwargs)
        clf.fit(train_features.cpu().numpy(), train_labels.cpu().numpy())
        pred = clf.predict(test_features.cpu().numpy())
        labels_np = test_labels.cpu().numpy()

        acc_frac = np.sum(labels_np == pred) / pred.shape[0]
        acc = acc_frac * 100.0 if return_percent else acc_frac

    print_log(f'  SVM validation: acc={acc_frac * 100:.2f}%', logger=logger)

    if val_writer is not None:
        val_writer.add_scalar(writer_tag, acc_frac * 100.0, epoch)

    return Acc_Metric(acc)
