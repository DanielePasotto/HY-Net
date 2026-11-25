from .baselines import UNet, AttentionUNet
from .hynet import HYNet

# name -> (description, uses a pre-trained HarmNet)
MODELS = {
    "hynet_res": ("HY-Net (final model): residual HarmNet (identity at start, no classical-pipeline target) + Asymmetric Gated Y-Net, trained end to end", False),
    "hynet_fast": ("hynet_res whose residual HarmNet works at half resolution (about 1/4 of its operations)", False),
    "hynet": ("HY-Net, pre-trained variant: HarmNet pre-trained on the classical-pipeline target, then fine-tuned end to end", True),
    "hynet_nogate": ("pre-trained variant with plain concatenation of the two streams, no attention and no gating", True),
    "ynet_raw": ("dual-stream Y-Net fed with the raw image in both streams (no HarmNet)", False),
    "unet": ("U-Net on the raw image", False),
    "attunet": ("Attention U-Net on the raw image", False),
    "campeod": ("CAMPEOD (third-party Swin-V2 network, clone it into CAMPEOD/) on the raw image", False),
    "macpnet": ("MACPnet (third-party MaxViT-Large network, clone it into MACPnet/) on the raw image", False),
}

def needs_harmnet(name):
    return MODELS[name][1]

def build_model(name, args, device, harmnet_weights_path=None):
    """Build one of the compared models.

    The models that start from a trained HarmNet load it from `harmnet_weights_path`; with None HarmNet is randomly
    initialized, which is what is wanted when a full checkpoint is loaded afterwards.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    shapes = (args.input_shape, args.hidden_units, args.output_shape)
    if name == "hynet_res":
        return HYNet(*shapes, device, None, residual_harm=True)
    if name == "hynet_fast":
        return HYNet(*shapes, device, None, residual_harm=True, harm_down=2)
    if name == "hynet":
        return HYNet(*shapes, device, harmnet_weights_path)
    if name == "hynet_nogate":
        return HYNet(*shapes, device, harmnet_weights_path, use_attention=False, fusion="concat")
    if name == "ynet_raw":
        return HYNet(*shapes, device, None, harmonize=False)
    if name == "campeod":
        from .campeod import Campeod  # imported here: the wrapper needs the third-party code, which is not in the repository
        return Campeod(args.input_shape, device)
    if name == "macpnet":
        from .macpnet import MACPNet  # same reason as CAMPEOD
        return MACPNet(args.input_shape, device)
    if name == "unet":
        return UNet(args.input_shape, args.output_shape).to(device)
    return AttentionUNet(args.input_shape, args.output_shape).to(device)
