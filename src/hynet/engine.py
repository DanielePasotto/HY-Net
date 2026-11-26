import os
import random
import numpy as np
import torch
from tqdm import tqdm

def seed_everything(seed):
    """Seed python, numpy and torch RNGs (the data augmentation uses python's `random`) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=20, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

        if mode == 'max':
            self.best_score = -float('inf')
        else:
            self.best_score = float('inf')

    def __call__(self, current_score):
        if self.mode == 'max':
            is_improvement = current_score > (self.best_score + self.min_delta)
        else:
            is_improvement = current_score < (self.best_score - self.min_delta)

        if is_improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def training(model, loader, criterion, optimizer, device, scaler=None, micro_batches=1):
    """One epoch. With a `torch.amp.GradScaler` the forward pass runs in fp16 autocast (the loss is always computed in fp32).

    With `micro_batches` > 1 every batch is split into that many chunks whose gradients are accumulated before a single optimizer step:
    the same batch size with a lower peak memory (BatchNorm statistics are computed per chunk).
    """
    model.train()
    running_loss = 0.0
    amp = scaler is not None

    for data in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        batch_loss = 0.0
        for chunk in zip(*(t.chunk(micro_batches) for t in data)):
            weight = len(chunk[0]) / len(data[0])
            inputs = chunk[0].to(device)
            masks = chunk[1].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(inputs)
            outputs = outputs.float()

            loss = criterion(outputs, masks)

            if amp:
                scaler.scale(loss * weight).backward()
            else:
                (loss * weight).backward()
            batch_loss += loss.item() * weight

        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running_loss += batch_loss

    return running_loss / len(loader)

def validation(model, loader, criterion, device, evaluator):
    """Mean loss and the metrics of `evaluator` (hynet.metrics.SemanticEvaluator) over the loader."""
    model.eval()
    evaluator.reset()
    segmentation_loss = 0.0
    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            inputs = data[0].to(device)
            masks = data[1].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, masks)
            segmentation_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
            evaluator.update(preds.squeeze(1), masks.long().squeeze(1))

    return {'loss': segmentation_loss / len(loader), **evaluator.compute()}
