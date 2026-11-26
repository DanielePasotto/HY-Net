import torch

class SemanticEvaluator:
    def __init__(self, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float, device=device)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask] + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape(self.num_classes, self.num_classes)

    def update(self, preds, targets):
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        self.confusion_matrix += self._generate_matrix(targets.flatten(), preds.flatten())

    def compute(self):
        cm = self.confusion_matrix

        TP = torch.diag(cm)
        FN = cm.sum(dim=1) - TP
        FP = cm.sum(dim=0) - TP

        precision_cls = TP / (TP + FP + 1e-6)
        recall_cls = TP / (TP + FN + 1e-6)

        iou_cls = TP / (TP + FP + FN + 1e-6)
        dice_cls = 2 * TP / (2 * TP + FP + FN + 1e-6)

        results = {
            "mIoU": iou_cls.mean().item(),
            "mDICE": dice_cls.mean().item(),
            "Precision": precision_cls.mean().item(),
            "Recall": recall_cls.mean().item(),
            "Global Accuracy": TP.sum().item() / cm.sum().item(),
            # class 1 = organoid: foreground-only scores (mDICE/mIoU above also average in the background class)
            "Organoid Dice": dice_cls[1].item(),
            "Organoid IoU": iou_cls[1].item(),

            "Class Precision": precision_cls.cpu().numpy().round(4),
            "Class Recall": recall_cls.cpu().numpy().round(4)
        }
        return results

    def reset(self):
        self.confusion_matrix.fill_(0)
