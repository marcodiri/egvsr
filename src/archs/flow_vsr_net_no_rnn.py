import torch
import torch.nn.functional as F

from archs.arch_utils import (
    BaseGenerator,
    backward_warp,
    get_upsampling_func,
    space_to_depth,
)
from archs.flow_net import FNet
from archs.sr_net import SRNet


class FRNet(BaseGenerator):
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nf=64,
        nb=16,
        degradation="BI",
        scale=4,
    ):
        super(FRNet, self).__init__()

        self.scale = scale

        # get upsampling function according to the degradation mode
        self.upsample_func = get_upsampling_func(self.scale, degradation)

        # define fnet & srnet
        self.fnet = FNet(in_nc)
        self.srnet = SRNet(2 * in_nc, out_nc, nf, nb, scale)

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_dummy_input(self, lr_size):
        c, lr_h, lr_w = lr_size
        s = self.scale

        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        hr_prev = torch.rand(1, c, s * lr_h, s * lr_w, dtype=torch.float32)

        data_dict = {"lr_curr": lr_curr, "lr_prev": lr_prev, "hr_prev": hr_prev}

        return data_dict

    def forward_single(self, lr_curr, lr_prev):
        """
        Parameters:
            :param lr_curr: the current lr data in shape nchw
            :param lr_prev: the previous lr data in shape nchw
        """

        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.fnet(lr_curr, lr_prev)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), "reflect")

        # warp hr_prev
        lr_prev_warp = backward_warp(lr_prev, lr_flow_pad)

        # compute hr_curr
        hr_curr = self.srnet(
            torch.cat(
                [
                    lr_curr,
                    lr_prev_warp,
                ],
                dim=1,
            )
        )

        return hr_curr

    def forward(self, lr_data):
        """
        Parameters:
            :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w

        lr_flow_v = lr_flow.view(n, (t - 1), 2, lr_h, lr_w)

        # compute the first hr data
        hr_data = []
        lr_prev_warp_data = []
        hr_prev = self.srnet(
            torch.cat(
                [
                    lr_data[:, 0, ...],
                    torch.zeros(
                        n,
                        c,
                        lr_h,
                        lr_w,
                        dtype=torch.float32,
                        device=lr_data.device,
                    ),
                ],
                dim=1,
            )
        )
        hr_data.append(hr_prev)
        lr_prev_warp_data.append(
            torch.zeros(
                n,
                c,
                lr_h,
                lr_w,
                dtype=torch.float32,
                device=lr_data.device,
            )
        )

        # compute the remaining hr data
        for i in range(1, t):
            # warp lr_prev
            lr_prev_warp = backward_warp(
                lr_data[:, i - 1, ...], lr_flow_v[:, i - 1, ...]
            )

            # compute hr_curr
            hr_curr = self.srnet(
                torch.cat(
                    [
                        lr_data[:, i, ...],
                        lr_prev_warp,
                    ],
                    dim=1,
                )
            )

            # save and update
            hr_data.append(hr_curr)
            lr_prev_warp_data.append(lr_prev_warp)

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w
        lr_prev_warp_data = torch.stack(lr_prev_warp_data, dim=1)  # n,t,c,hr_h,hr_w

        # construct output dict
        ret_dict = {
            "hr_data": hr_data,  # n,t,c,hr_h,hr_w
            "lr_prev": lr_prev,  # n(t-1),c,lr_h,lr_w
            "lr_curr": lr_curr,  # n(t-1),c,lr_h,lr_w
            "lr_flow": lr_flow,  # n(t-1),2,lr_h,lr_w
            "lr_prev_warp": lr_prev_warp_data,
        }

        return ret_dict

    def infer_sequence(self, lr_data):
        """
        Parameters:
            :param lr_data: torch.FloatTensor in shape tchw

            :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # setup params
        tot_frm, c, h, w = lr_data.size()

        # forward
        hr_seq = []
        lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for i in range(tot_frm):
                lr_curr = lr_data[i : i + 1, ...]
                hr_curr = self.forward_single(lr_curr, lr_prev)
                lr_prev = lr_curr

                hr_frm = hr_curr.squeeze(0).cpu()  # chw|rgb|uint8

                hr_seq.append(hr_frm)

        return torch.stack(hr_seq)
