import torch
from skimage.filters import threshold_otsu

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

def calculate_harmonization_metrics(pred, mask):
    with torch.no_grad():
        pred_np = pred.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        try:
            thresh = threshold_otsu(pred_np)
            binary_pred = pred_np > thresh

            if binary_pred.sum() > (binary_pred.size / 2):
                binary_pred = ~binary_pred

            intersection = (binary_pred * mask_np).sum()
            union = binary_pred.sum() + mask_np.sum()
            dice = (2. * intersection + 1e-6) / (union + 1e-6)

        except Exception:
            dice = 0.0

        pred_flat = pred.flatten()
        mask_flat = mask.flatten()

        cell_pixels = pred_flat[mask_flat > 0.5]
        bg_pixels = pred_flat[mask_flat <= 0.5]

        if len(cell_pixels) == 0 or len(bg_pixels) == 0:
            return {'cnr': 0.0, 'bg_roughness': 0.0, 'dice': dice}

        mu_cell = torch.mean(cell_pixels)
        mu_bg = torch.mean(bg_pixels)
        std_bg = torch.std(bg_pixels)

        bg_roughness = std_bg.item()

        cnr = torch.abs(mu_cell - mu_bg) / (std_bg + 1e-6)

    return {
        'cnr': cnr.item(),
        'bg_roughness': bg_roughness,
        'dice': dice
    }
