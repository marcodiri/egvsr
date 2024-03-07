from typing import Dict

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lpips import LPIPS

from archs.arch_utils import BaseGenerator, flow_warp
from optim import define_criterion
from optim.losses import SSIM


class VSRSingle(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        *,
        losses: Dict,
        gen_lr: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator"])
        self.G = generator

        # pixel criterion
        self.pix_crit, self.pix_w = define_criterion(losses.get("pixel_crit"))

        # warping criterion
        self.warp_crit, self.warp_w = define_criterion(losses.get("warping_crit"))

        # ping-pong criterion
        self.pp_crit, self.pp_w = define_criterion(losses.get("pingpong_crit"))

        # feature criterion
        self.feat_crit, self.feat_w = define_criterion(losses.get("feature_crit"))

        # validation losses
        self.lpips_alex = LPIPS(net="alex", version="0.1")
        self.ssim = SSIM()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim_G = torch.optim.Adam(params=self.G.parameters(), lr=self.hparams.gen_lr)

        return optim_G

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # ------------ prepare data ------------ #
        gt_data, lr_data = batch

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        assert t > 1, "A temporal radius of at least 2 is needed"

        # augment data for pingpong criterion
        if self.pp_crit is not None:
            # i.e., (0,1,2,...,t-2,t-1) -> (0,1,2,...,t-2,t-1,t-2,...,2,1,0)
            lr_rev = lr_data.flip(1)[:, 1:, ...]
            gt_rev = gt_data.flip(1)[:, 1:, ...]

            lr_data = torch.cat([lr_data, lr_rev], dim=1)
            gt_data = torch.cat([gt_data, gt_rev], dim=1)

        to_log, to_log_prog = {}, {}

        # ------------ forward G ------------ #
        net_G_output_dict = self.G(lr_data)
        hr_data = net_G_output_dict["hr_data"]

        # ------------ optimize G ------------ #

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            loss_pix_G = self.pix_crit(hr_data, gt_data)
            loss_G += self.pix_w * loss_pix_G
            to_log["G_pixel_loss"] = loss_pix_G

        # warping (warp) loss
        if self.warp_crit is not None:
            lr_curr = net_G_output_dict["lr_curr"]
            lr_prev = net_G_output_dict["lr_prev"]
            lr_flow = net_G_output_dict["lr_flow"]
            lr_warp = flow_warp(lr_prev, lr_flow.permute(0, 2, 3, 1))

            loss_warp_G = self.warp_crit(lr_warp, lr_curr)
            loss_G += self.warp_w * loss_warp_G
            to_log["G_warping_loss"] = loss_warp_G

        # feature (feat) loss
        if self.feat_crit is not None:
            hr_merge = hr_data.view(-1, c, gt_h, gt_w)
            gt_merge = gt_data.view(-1, c, gt_h, gt_w)
            loss_feat_G = self.feat_crit(hr_merge, gt_merge).mean()

            loss_G += self.feat_w * loss_feat_G
            to_log_prog["G_lpip_loss"] = loss_feat_G

        # ping-pong (pp) loss
        if self.pp_crit is not None:
            hr_data_fw = hr_data[:, : t - 1, ...]  #    -------->|
            hr_data_bw = hr_data[:, t:, ...].flip(1)  # <--------|

            loss_pp_G = self.pp_crit(hr_data_fw, hr_data_bw)
            loss_G += self.pp_w * loss_pp_G
            to_log["G_ping_pong_loss"] = loss_pp_G

        self.log_dict(to_log_prog, prog_bar=True)
        self.log_dict(to_log, prog_bar=False)

        return loss_G

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        gt_data, lr_data = batch
        _, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        net_G_output_dict = self.G(lr_data)
        hr_data = net_G_output_dict["hr_data"]

        hr_merge = hr_data.view(-1, c, gt_h, gt_w)
        gt_merge = gt_data.view(-1, c, gt_h, gt_w)

        ssim_val = self.ssim(hr_merge, gt_merge).mean()
        lpips_val = self.lpips_alex(hr_merge, gt_merge).mean()

        self.log_dict(
            {
                "val_ssim": ssim_val,
                "val_lpips": lpips_val,
            },
            on_epoch=True,
            prog_bar=True,
        )

        # lr_curr = net_G_output_dict["lr_curr"]
        # lr_prev = net_G_output_dict["lr_prev"]
        # lr_flow = net_G_output_dict["lr_flow"]
        # lr_warp = backward_warp(lr_prev, lr_flow)

        return (
            (
                lr_data.view(-1, c, lr_h, lr_w),
                (
                    F.interpolate(
                        (net_G_output_dict["hr_prev_warp"].view(-1, c, gt_h, gt_w)),
                        size=lr_data.shape[-2:],
                        mode="bicubic",
                    )
                    if "hr_prev_warp" in net_G_output_dict
                    else net_G_output_dict["lr_prev_warp"].view(-1, c, lr_h, lr_w)
                ),
                F.interpolate(
                    gt_data.view(-1, c, gt_h, gt_w),
                    size=lr_data.shape[-2:],
                    mode="bicubic",
                ),
            ),
            (
                gt_merge,
                hr_merge,
                F.interpolate(
                    lr_data.view(-1, c, lr_h, lr_w),
                    size=gt_data.shape[-2:],
                    mode="bicubic",
                ),
            ),
            ("lq vs aligned vs hq downscaled", "hq vs fake vs bicubic"),
        )
