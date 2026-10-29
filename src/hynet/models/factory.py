from .baselines import UNet, AttentionUNet, HarmUNet
from .hynet import HYNet

# name -> (description, uses a pre-trained HarmNet)
MODELS = {
    "hynet": ("full HY-Net: HarmNet (fine-tuned end to end) + dual-stream gated Y-Net", True),
    "hynet_frozen": ("HY-Net with a frozen HarmNet (static pre-processing)", True),
    "hynet_nogate": ("HY-Net with plain concatenation of the two streams, no attention and no gating", True),
    "ynet_raw": ("dual-stream Y-Net fed with the raw image in both streams (no harmonization)", False),
    "harm_unet": ("HarmNet + single-stream U-Net on the harmonized image", True),
    "unet": ("U-Net on the raw image", False),
    "attunet": ("Attention U-Net on the raw image", False),
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
    if name == "hynet":
        return HYNet(*shapes, device, harmnet_weights_path)
    if name == "hynet_frozen":
        return HYNet(*shapes, device, harmnet_weights_path, freeze_harmnet=True)
    if name == "hynet_nogate":
        return HYNet(*shapes, device, harmnet_weights_path, use_attention=False, fusion="concat")
    if name == "ynet_raw":
        return HYNet(*shapes, device, None, harmonize=False)
    if name == "harm_unet":
        return HarmUNet(*shapes, device, harmnet_weights_path)
    if name == "unet":
        return UNet(args.input_shape, args.output_shape).to(device)
    return AttentionUNet(args.input_shape, args.output_shape).to(device)
