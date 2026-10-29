import time
import torch
from hynet.config import get_args, str2bool
from hynet.losses import TverskyBCELoss
from hynet.models.factory import MODELS, build_model

def add_arguments(parser):
    group = parser.add_argument_group("benchmark")
    group.add_argument("--models", type=str, nargs="+", default=list(MODELS), help="Models to benchmark")
    group.add_argument("--iters", type=int, default=5, help="Timed training iterations per model")
    group.add_argument("--train_images", type=int, default=195, help="Training images per fold, to extrapolate the time of an epoch")
    group.add_argument("--inference", type=str2bool, default=False, help="Measure parameters, latency and peak memory of inference instead of training")

@torch.no_grad()
def benchmark_inference(args, device):
    """Parameters, latency per image (batch 1, 480x640, random input) and peak memory of the forward pass in eval mode, fp32 and fp16."""
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}, batch size 1, {args.iters} timed forward passes")
    print(f"{'model':<14}{'params (M)':>11}{'fp32 ms':>9}{'fp32 GB':>9}{'fp16 ms':>9}{'fp16 GB':>9}")
    for name in args.models:
        model = build_model(name, args, device).eval()
        x = torch.rand(1, 1, 480, 640, device=device)
        row = []
        for dtype in (None, torch.float16):
            torch.cuda.reset_peak_memory_stats()
            for i in range(args.iters + 5):  # the first five passes are warm-up
                if i == 5:
                    torch.cuda.synchronize()
                    start = time.time()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dtype is not None):
                    model(x)
            torch.cuda.synchronize()
            row += [(time.time() - start) / args.iters * 1000, torch.cuda.max_memory_allocated() / 1e9]
        print(f"{name:<14}{sum(p.numel() for p in model.parameters()) / 1e6:>11.1f}{row[0]:>9.1f}{row[1]:>9.2f}{row[2]:>9.1f}{row[3]:>9.2f}")
        model = x = None
        torch.cuda.empty_cache()

def main():
    """Peak GPU memory and speed of one training iteration (random data, batch size --batch_size, 480x640) of each model."""
    args = get_args(add_arguments)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.inference:
        benchmark_inference(args, device)
        return
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}, batch size {args.batch_size}, amp {args.amp}")
    print(f"{'model':<14}{'params (M)':>11}{'peak mem (GB)':>15}{'s/iter':>9}{'min/epoch':>11}")
    criterion = TverskyBCELoss(args.alpha, args.beta).to(device)
    for name in args.models:
        try:
            model = build_model(name, args, device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
            x = torch.rand(args.batch_size, 1, 480, 640, device=device)
            y = (torch.rand(args.batch_size, 1, 480, 640, device=device) > 0.8).float()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for i in range(args.iters + 2):  # the first two iterations are warm-up
                if i == 2:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.time()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
                    out = model(x)
                loss = criterion(out.float(), y)
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            per_iter = (time.time() - start) / args.iters
            peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
            epoch_min = per_iter * (args.train_images / args.batch_size) / 60
            print(f"{name:<14}{sum(p.numel() for p in model.parameters()) / 1e6:>11.1f}{peak:>15.2f}{per_iter:>9.2f}{epoch_min:>11.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{name:<14}{'OUT OF MEMORY':>46}")
        finally:
            model = optimizer = x = y = loss = out = None
            if device.type == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
