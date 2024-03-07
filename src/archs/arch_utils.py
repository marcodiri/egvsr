import functools
from collections import namedtuple
from typing import Iterable, List, Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models as tv


class BaseGenerator(L.LightningModule):
    pass


class BaseDiscriminator(L.LightningModule):
    pass


def space_to_depth(x, scale=4):
    """Equivalent to tf.space_to_depth()"""

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output


def backward_warp(x, flow, mode="bilinear", padding_mode="border"):
    """Backward warp `x` according to `flow`

    Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

    Reference:
        https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat(
        [flow[:, 0:1, ...] / ((w - 1.0) / 2.0), flow[:, 1:2, ...] / ((h - 1.0) / 2.0)],
        dim=1,
    )

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default in PyTorch version
    #        lower than 1.4.0
    if int("".join(torch.__version__.split(".")[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True
        )
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


def flow_warp(
    x, flow, interp_mode="bilinear", padding_mode="zeros", align_corners=True
):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)
    )
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # TODO, what if align_corners=False
    return output


# def flow_warp(
#     x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True
# ):
#     """Warp an image or a feature map with optical flow.

#     Args:
#         x (Tensor): Tensor with size (n, c, h, w).
#         flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
#             a two-channel, denoting the width and height relative offsets.
#             Note that the values are not normalized to [-1, 1].
#         interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
#             Default: 'bilinear'.
#         padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
#             Default: 'zeros'.
#         align_corners (bool): Whether align corners. Default: True.

#     Returns:
#         Tensor: Warped image or feature map.
#     """
#     if x.size()[-2:] != flow.size()[1:3]:
#         raise ValueError(
#             f"The spatial sizes of input ({x.size()[-2:]}) and "
#             f"flow ({flow.size()[1:3]}) are not the same."
#         )
#     _, _, h, w = x.size()
#     # create mesh grid
#     device = flow.device
#     # torch.meshgrid has been modified in 1.10.0 (compatibility with previous
#     # versions), and will be further modified in 1.12 (Breaking Change)
#     if "indexing" in torch.meshgrid.__code__.co_varnames:
#         grid_y, grid_x = torch.meshgrid(
#             torch.arange(0, h, device=device, dtype=x.dtype),
#             torch.arange(0, w, device=device, dtype=x.dtype),
#             indexing="ij",
#         )
#     else:
#         grid_y, grid_x = torch.meshgrid(
#             torch.arange(0, h, device=device, dtype=x.dtype),
#             torch.arange(0, w, device=device, dtype=x.dtype),
#         )
#     grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
#     grid.requires_grad = False

#     grid_flow = grid + flow
#     # scale grid_flow to [-1,1]
#     grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
#     grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
#     grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
#     grid_flow = grid_flow.type(x.type())
#     output = F.grid_sample(
#         x,
#         grid_flow,
#         mode=interpolation,
#         padding_mode=padding_mode,
#         align_corners=align_corners,
#     )
#     return output


def get_upsampling_func(scale=4, degradation="BI"):
    if degradation == "BI":
        upsample_func = functools.partial(
            F.interpolate, scale_factor=scale, mode="bilinear", align_corners=False
        )

    elif degradation == "BD":
        upsample_func = BicubicUpsample(scale_factor=scale)

    else:
        raise ValueError("Unrecognized degradation: {}".format(degradation))

    return upsample_func


class BicubicUpsample(nn.Module):
    """A bicubic upsampling class with similar behavior to that in TecoGAN-Tensorflow

    Note that it's different from torch.nn.functional.interpolate and
    matlab's imresize in terms of bicubic kernel and sampling scheme

    Theoretically it can support any scale_factor >= 1, but currently only
    scale_factor = 4 is tested

    References:
        The original paper: http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
        https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()

        # calculate weights
        cubic = torch.FloatTensor(
            [
                [0, a, -2 * a, a],
                [1, 0, -(a + 3), a + 2],
                [0, -a, (2 * a + 3), -(a + 2)],
                [0, 0, a, -a],
            ]
        )  # accord to Eq.(6) in the reference paper

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s**2, s**3]))
            for s in [1.0 * d / scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer("kernels", torch.stack(kernels))

    def forward(self, input):
        n, c, h, w = input.size()
        s = self.scale_factor

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode="replicate")

        # calculate output (height)
        kernel_h = self.kernels.repeat(c, 1).view(-1, 1, s, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0, groups=c)
        output = (
            output.reshape(n, c, s, -1, w + 3)
            .permute(0, 1, 3, 2, 4)
            .reshape(n, c, -1, w + 3)
        )

        # calculate output (width)
        kernel_w = self.kernels.repeat(c, 1).view(-1, 1, 1, s)
        output = F.conv2d(output, kernel_w, stride=1, padding=0, groups=c)
        output = (
            output.reshape(n, c, s, h * s, -1)
            .permute(0, 1, 3, 4, 2)
            .reshape(n, c, h * s, -1)
        )

        return output


class Conv3Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
    ):
        super().__init__()

        self.block1 = Conv3Block(dim, dim_out)
        self.block2 = Conv3Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        return h + self.res_conv(x)


class ResNet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(ResNet, self).__init__()
        self.net = tv.resnet18(weights="DEFAULT")

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

        if not requires_grad:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ["relu1", "conv2", "conv3", "conv4", "conv5"])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out
