from .baselines import UNet, AttentionUNet
from .hynet import HYNet

# name -> description
MODELS = {
    "hynet_res": "HY-Net (final model): residual HarmNet (identity at start, trained only with the segmentation loss) + Asymmetric Gated Y-Net",
    "hynet_fast": "hynet_res whose residual HarmNet works at half resolution (about 1/4 of its operations)",
    "hynet_res_nogate": "hynet_res with plain concatenation of the two streams, no attention and no gating (ablation)",
    "ynet_raw": "dual-stream Y-Net fed with the raw image in both streams (no HarmNet)",
    "unet": "U-Net on the raw image",
    "attunet": "Attention U-Net on the raw image",
    "campeod": "CAMPEOD (third-party Swin-V2 network, clone it into CAMPEOD/) on the raw image",
    "macpnet": "MACPnet (third-party MaxViT-Large network, clone it into MACPnet/) on the raw image",
}

def build_model(name, args, device):
    """Build one of the compared models (randomly initialized)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    shapes = (args.input_shape, args.hidden_units, args.output_shape)
    if name == "hynet_res":
        return HYNet(*shapes, device)
    if name == "hynet_fast":
        return HYNet(*shapes, device, harm_down=2)
    if name == "hynet_res_nogate":
        return HYNet(*shapes, device, use_attention=False, fusion="concat")
    if name == "ynet_raw":
        return HYNet(*shapes, device, harmonize=False)
    if name == "campeod":
        from .campeod import Campeod  # imported here: the wrapper needs the third-party code, which is not in the repository
        return Campeod(args.input_shape, device)
    if name == "macpnet":
        from .macpnet import MACPNet  # same reason as CAMPEOD
        return MACPNet(args.input_shape, device)
    if name == "unet":
        return UNet(args.input_shape, args.output_shape).to(device)
    return AttentionUNet(args.input_shape, args.output_shape).to(device)
