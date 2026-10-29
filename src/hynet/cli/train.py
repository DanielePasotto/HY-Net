import os
import torch
from timeit import default_timer as timer
from hynet.metrics import SemanticEvaluator
from hynet.losses import TverskyBCELoss, RegionAwareCompositeLoss
from hynet.models.factory import build_model, needs_harmnet
from hynet.models.harmnet import HarmNet
from torch import optim
from hynet.engine import EarlyStopping, training, validation, seed_everything
from hynet.splits import segmentation_dataset, harmonization_dataset, weights_paths
from torch.utils.data import DataLoader
from hynet.config import get_args

def train_segmentation(args, model, train_loader, val_loader, device, weights_path):
    """Train a segmentation model (any of hynet.models.factory.MODELS) with the Tversky-BCE loss; the best validation mDICE is kept."""
    if args.fine_tune:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print("### Model loaded for fine-tuning ###")
    evaluator = SemanticEvaluator(num_classes=args.num_classes, device=device)
    criterion = TverskyBCELoss(args.alpha, args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, args.scheduler_mode, patience=args.scheduler_patience)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode=args.early_stopping_mode)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    best_val_dice = float('-inf')

    print(f"Starting training of '{args.model}' on {device}...")

    start_time = timer()
    for epoch in range(args.epochs):
        train_loss = training(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validation(model, val_loader, criterion, device, evaluator)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Global Accuracy: {val_metrics['Global Accuracy']:.2f} | "
            f"mIoU: {val_metrics['mIoU']:.3f} | "
            f"mDICE: {val_metrics['mDICE']:.3f} | "
            f"Organoid Dice: {val_metrics['Organoid Dice']:.3f} | "
            f"Precision: {val_metrics['Precision']:.3f} | "
            f"Recall: {val_metrics['Recall']:.3f}")

        if val_metrics['mDICE'] > best_val_dice:
            best_val_dice = val_metrics['mDICE']
            torch.save(model.state_dict(), weights_path)
            print("--> Model saved (Best Val mDICE)")

        scheduler.step(val_metrics['mDICE'])
        early_stopping(val_metrics['mDICE'])

        if early_stopping.early_stop:
            print("\n" + "#"*30)
            print(f"EARLY STOPPING TRIGGERED at Epoch {epoch+1}")
            print(f"The model did not improve mDICE for {early_stopping.patience} epochs.")
            print("#"*30)
            break

    end_time = timer()
    total_time = end_time - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")

def train_harmnet(args, train_loader, val_loader, device, weights_path):
    model = HarmNet(input_shape=args.input_shape, hidden_units=args.hidden_units, output_shape=args.output_shape).to(device)
    criterion = RegionAwareCompositeLoss(lambda_mae=args.lambda_mae, lambda_ssim=args.lambda_ssim, lambda_tv=args.lambda_tv, lambda_edge=args.lambda_edge, lambda_bg=args.lambda_bg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, args.scheduler_mode, patience=args.scheduler_patience)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode=args.early_stopping_mode)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    best_val_ssim = float('-inf')

    print(f"Starting training on {device}...")

    start_time = timer()
    for epoch in range(args.epochs):
        train_loss = training(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validation(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"PSNR: {val_metrics['psnr']:.2f} | "
            f"SSIM: {val_metrics['ssim']:.3f} | "
            f"CNR: {val_metrics['cnr']:.2f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"BG Roughness: {val_metrics['bg_roughness']:.4f}")

        if val_metrics['ssim'] > best_val_ssim:
            best_val_ssim = val_metrics['ssim']
            torch.save(model.state_dict(), weights_path)
            print("--> Model saved (Best Val SSIM)")

        scheduler.step(val_metrics['ssim'])

        early_stopping(val_metrics['ssim'])

        if early_stopping.early_stop:
            print("\n" + "!"*30)
            print(f"EARLY STOPPING TRIGGERED at Epoch {epoch+1}")
            print(f"The model did not improve SSIM for {early_stopping.patience} epochs.")
            print("!"*30)
            break

    end_time = timer()
    total_time = end_time - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    harmnet_path, model_path = weights_paths(args)
    for weights_path in (harmnet_path, model_path):
        os.makedirs(os.path.dirname(weights_path) or '.', exist_ok=True)

    seed_everything(args.seed)

    if args.harmnet or args.both:
        train_loader = DataLoader(harmonization_dataset(args, "train"), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(harmonization_dataset(args, "val"), batch_size=1, shuffle=False)
        train_harmnet(args, train_loader, val_loader, device, harmnet_path)
    if args.hynet or args.both:
        model = build_model(args.model, args, device, harmnet_path if needs_harmnet(args.model) else None)
        train_loader = DataLoader(segmentation_dataset(args, "train", train=True), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(segmentation_dataset(args, "val"), batch_size=1, shuffle=False)
        train_segmentation(args, model, train_loader, val_loader, device, model_path)

if __name__ == "__main__":
    main()
