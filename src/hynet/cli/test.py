import torch
from timeit import default_timer as timer
from hynet.metrics import SemanticEvaluator
from hynet.losses import TverskyBCELoss, RegionAwareCompositeLoss
from hynet.models.factory import build_model
from hynet.models.harmnet import HarmNet
from hynet.engine import validation, seed_everything
from hynet.splits import segmentation_dataset, harmonization_dataset, weights_paths
from torch.utils.data import DataLoader
from hynet.config import get_args

def test_segmentation(args, test_loader, device, weights_path):
    model = build_model(args.model, args, device)  # weights come from the checkpoint
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print("##### Model loaded correctly #####")
    evaluator = SemanticEvaluator(num_classes=args.num_classes, device=device)
    criterion = TverskyBCELoss(args.alpha, args.beta).to(device)

    print(f"Starting testing of '{args.model}' on {device}...")

    start_time = timer()
    test_metrics = validation(model, test_loader, criterion, device, evaluator)

    print(f"Test Metrics: \n"
        f"Loss: {test_metrics['loss']:.4f} \n"
        f"Global Accuracy: {test_metrics['Global Accuracy']:.2f} \n"
        f"mIoU: {test_metrics['mIoU']:.3f} \n"
        f"mDICE: {test_metrics['mDICE']:.3f} \n"
        f"Organoid Dice (foreground only): {test_metrics['Organoid Dice']:.3f} \n"
        f"Organoid IoU (foreground only): {test_metrics['Organoid IoU']:.3f} \n"
        f"Precision: {test_metrics['Precision']:.3f} \n"
        f"Recall: {test_metrics['Recall']:.3f} \n"
        f"{30 * '-'} \n"
        f"Class-wise Precision: {test_metrics['Class Precision']} \n"
        f"Class-wise Recall: {test_metrics['Class Recall']}")

    end_time = timer()
    total_time = end_time - start_time
    print(f"Inference completed in {total_time:.2f} seconds.")

def test_harmnet(args, test_loader, device, weights_path):
    model = HarmNet(input_shape=args.input_shape, hidden_units=args.hidden_units, output_shape=args.output_shape).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print("##### Model loaded correctly #####")
    criterion = RegionAwareCompositeLoss(lambda_mae=args.lambda_mae, lambda_ssim=args.lambda_ssim, lambda_tv=args.lambda_tv, lambda_edge=args.lambda_edge, lambda_bg=args.lambda_bg).to(device)

    print(f"Starting testing on {device}...")

    start_time = timer()
    test_metrics = validation(model, test_loader, criterion, device)

    print(f"Test Metrics: \n"
        f"Loss: {test_metrics['loss']:.4f} \n"
        f"PSNR: {test_metrics['psnr']:.2f} \n"
        f"SSIM: {test_metrics['ssim']:.3f} \n"
        f"CNR: {test_metrics['cnr']:.2f} \n"
        f"Dice: {test_metrics['dice']:.4f} \n"
        f"BG Roughness: {test_metrics['bg_roughness']:.4f}")

    end_time = timer()
    total_time = end_time - start_time
    print(f"Inference completed in {total_time:.2f} seconds.")

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    harmnet_path, model_path = weights_paths(args)

    seed_everything(args.seed)

    if args.harmnet or args.both:
        test_loader = DataLoader(harmonization_dataset(args, "test"), batch_size=1, shuffle=False)
        test_harmnet(args, test_loader, device, harmnet_path)

    if args.hynet or args.both:
        test_loader = DataLoader(segmentation_dataset(args, "test"), batch_size=1, shuffle=False)
        test_segmentation(args, test_loader, device, model_path)

if __name__ == "__main__":
    main()
