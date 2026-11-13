import torch
import torch.nn as nn

def adapt_batchnorm(model, images, device):
    """Adaptive BatchNorm: re-estimate the running statistics of every BatchNorm layer on `images` (an iterable of input batches, no labels needed).

    The weights are not changed; the statistics of the training domain are replaced by the average of the batch statistics of the target images.
    """
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the images
    model.eval()
    for m in norms:
        m.train()
    with torch.no_grad():
        for image in images:
            model(image.to(device))
    model.eval()
    return len(norms)
