"""Gatys et al. (2016) optimization-based style transfer with torchvision VGG-19.

Style loss on relu1_1..relu5_1 Gram matrices, content loss on relu4_2, L-BFGS.
torchvision weights are used instead of the original Caffe port, so feature scales
differ and the style weight lambda is not numerically comparable to other work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torchvision import models, transforms

from rbb.config import IMG_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STYLE_LAYERS = ["r11", "r21", "r31", "r41", "r51"]
CONTENT_LAYERS = ["r42"]
STYLE_LAYER_CHANNELS = [64, 128, 256, 512, 512]
# Per-layer normalization of the style loss, without the lambda factor.
STYLE_LAYER_WEIGHTS = [1e3 / n**2 for n in STYLE_LAYER_CHANNELS]
MAX_ITER = 500

_CONV_NAMES = [
    "conv1_1", "conv1_2", "conv2_1", "conv2_2",
    "conv3_1", "conv3_2", "conv3_3", "conv3_4",
    "conv4_1", "conv4_2", "conv4_3", "conv4_4",
    "conv5_1", "conv5_2", "conv5_3", "conv5_4",
]
_CONV_IDX = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
_CHANNELS = [(3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 256), (256, 256),
             (256, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512)]


class VGG(nn.Module):
    """VGG-19 conv stack exposing intermediate ReLU activations by name."""

    def __init__(self):
        super().__init__()
        for name, (c_in, c_out) in zip(_CONV_NAMES, _CHANNELS):
            setattr(self, name, nn.Conv2d(c_in, c_out, 3, padding=1))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, out_keys):
        out = {}
        out["r11"] = F.relu(self.conv1_1(x))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool(out["r12"])
        out["r21"] = F.relu(self.conv2_1(out["p1"]))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool(out["r22"])
        out["r31"] = F.relu(self.conv3_1(out["p2"]))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["p3"] = self.pool(out["r34"])
        out["r41"] = F.relu(self.conv4_1(out["p3"]))
        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["p4"] = self.pool(out["r44"])
        out["r51"] = F.relu(self.conv5_1(out["p4"]))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        return [out[k] for k in out_keys]


def build_vgg(device) -> VGG:
    vgg = VGG()
    tv = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    for name, idx in zip(_CONV_NAMES, _CONV_IDX):
        dst = getattr(vgg, name)
        dst.weight.data.copy_(tv[idx].weight.data)
        dst.bias.data.copy_(tv[idx].bias.data)
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg.to(device).eval()


class GramMatrix(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)).div(h * w)


class GramMSELoss(nn.Module):
    def forward(self, x, target):
        return nn.MSELoss()(GramMatrix()(x), target)


def style_transfer(vgg, content_img, style_img, style_weight_scale, max_iter=MAX_ITER, device="cuda"):
    loss_layers = STYLE_LAYERS + CONTENT_LAYERS
    loss_fns = [GramMSELoss().to(device)] * len(STYLE_LAYERS) + [nn.MSELoss().to(device)] * len(CONTENT_LAYERS)
    # Same operation order as the original generator (float-exact).
    weights = [style_weight_scale * 1e3 / n**2 for n in STYLE_LAYER_CHANNELS] + [1.0]

    style_targets = [GramMatrix()(a).detach() for a in vgg(style_img, STYLE_LAYERS)]
    content_targets = [a.detach() for a in vgg(content_img, CONTENT_LAYERS)]
    targets = style_targets + content_targets

    opt_img = content_img.clone().requires_grad_(True)
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]

    while n_iter[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            loss = sum(weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out))
            loss.backward()
            n_iter[0] += 1
            return loss
        optimizer.step(closure)

    return opt_img.detach()


resize_crop = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE)])
normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
prep = transforms.Compose([resize_crop, normalize])
_inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1 / s for s in IMAGENET_STD],
)


def postprocess(tensor) -> Image.Image:
    return transforms.ToPILImage()(_inv_normalize(tensor.clone()).clamp(0, 1))


def load_prepped(path, device) -> torch.Tensor:
    return prep(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
