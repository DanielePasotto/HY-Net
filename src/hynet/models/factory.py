from .baselines import UNet, AttentionUNet, HarmUNet
from .hynet import HYNet

# name -> (description, uses a pre-trained HarmNet)
MODELS = {
    "hynet": ("full HY-Net: HarmNet (fine-tuned end to end) + dual-stream gated Y-Net", True),
    "hynet_frozen": ("HY-Net with a frozen HarmNet (static pre-processing)", True),
    "hynet_decoder": ("HY-Net with frozen HarmNet and encoders; adapt fusion and decoder", True),
    "hynet_sum": ("HY-Net with attention and additive fusion instead of learned gates", True),
    "hynet_nogate": ("HY-Net with plain concatenation of the two streams, no attention and no gating", True),
    "hynet_res": ("HY-Net with a residual HarmNet (identity at start, no classical-pipeline target), trained end to end", False),
    "hynet_fast": ("hynet_res whose residual HarmNet works at half resolution (about 1/4 of its operations)", False),
    "hynet_par": ("HY-Net with a parametric harmonizer (smooth illumination field + monotone tone curve), trained end to end", False),
    "ynet_raw": ("dual-stream Y-Net fed with the raw image in both streams (no harmonization)", False),
    "harm_unet": ("HarmNet + single-stream U-Net on the harmonized image", True),
    "unet": ("U-Net on the raw image", False),
    "attunet": ("Attention U-Net on the raw image", False),
    "campeod": ("CAMPEOD (third-party Swin-V2 network, clone it into CAMPEOD/) on the raw image", False),
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
    norm = getattr(args, "norm", "batch")  # normalization of the Y-Net layers (HarmNet always uses InstanceNorm)
    if name == "hynet":
        return HYNet(*shapes, device, harmnet_weights_path, norm=norm)
    if name == "hynet_frozen":
        return HYNet(*shapes, device, harmnet_weights_path, freeze_harmnet=True, norm=norm)
    if name == "hynet_decoder":
        return HYNet(*shapes, device, harmnet_weights_path, freeze_harmnet=True, freeze_encoders=True, norm=norm)
    if name == "hynet_sum":
        return HYNet(*shapes, device, harmnet_weights_path, fusion="sum", norm=norm)
    if name == "hynet_nogate":
        return HYNet(*shapes, device, harmnet_weights_path, use_attention=False, fusion="concat", norm=norm)
    if name == "hynet_res":
        return HYNet(*shapes, device, None, norm=norm, residual_harm=True)
    if name == "hynet_fast":
        return HYNet(*shapes, device, None, norm=norm, residual_harm=True, harm_down=2)
    if name == "hynet_par":
        return HYNet(*shapes, device, None, norm=norm, param_harm=True)
    if name == "ynet_raw":
        return HYNet(*shapes, device, None, harmonize=False, norm=norm)
    if name == "harm_unet":
        return HarmUNet(*shapes, device, harmnet_weights_path)
    if name == "campeod":
        from .campeod import Campeod  # imported here: the wrapper needs the third-party code, which is not in the repository
        return Campeod(args.input_shape, device)
    if name == "unet":
        return UNet(args.input_shape, args.output_shape).to(device)
    return AttentionUNet(args.input_shape, args.output_shape).to(device)
