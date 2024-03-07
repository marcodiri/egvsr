from typing import Dict

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lpips import LPIPS

from archs.arch_utils import BaseGenerator, backward_warp, flow_warp
from optim import define_criterion
from optim.losses import SSIM, CharbonnierLoss


class AlignModule(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        *,
        losses: Dict,
        upscale_factor,
        gen_lr: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator"])
        self.G = generator

        # pixel criterion
        self.pix_crit, self.pix_w = define_criterion(losses.get("pixel_crit"))

        # feature criterion
        self.feat_crit, self.feat_w = define_criterion(losses.get("feature_crit"))

        # validation losses
        self.pix_crit_val = CharbonnierLoss(reduction="mean")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim_G = torch.optim.Adam(params=self.G.parameters(), lr=self.hparams.gen_lr)

        return optim_G

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # ------------ prepare data ------------ #
        gt_data = batch
        # lr_data = gt_data

        b, t, c, gt_h, gt_w = gt_data.size()

        assert t == 2, "A temporal radius of 2 is needed"

        to_log, to_log_prog = {}, {}

        # ------------ forward G ------------ #
        # estimate flow (curr -> prev)
        flow = self.G(gt_data[:, 1], gt_data[:, 0])
        # aligned = backward_warp(gt_data[:, 0], flow)
        aligned = flow_warp(gt_data[:, 0], flow.permute(0, 2, 3, 1))

        # ------------ optimize G ------------ #

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            loss_pix_G = self.pix_crit(
                aligned,
                gt_data[:, 1],
            )
            loss_G += self.pix_w * loss_pix_G
            to_log_prog["G_pixel_loss"] = loss_pix_G

        # feature (feat) loss
        if self.feat_crit is not None:
            loss_feat_G = self.feat_crit(
                aligned,
                gt_data[:, 1],
            ).mean()
            loss_G += self.feat_w * loss_feat_G
            to_log_prog["G_lpip_loss"] = loss_feat_G

        self.log_dict(to_log_prog, prog_bar=True)
        self.log_dict(to_log, prog_bar=False)

        return loss_G

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        gt_data = batch
        # lr_data = gt_data

        _, t, c, gt_h, gt_w = gt_data.size()

        flow = self.G(gt_data[:, 1], gt_data[:, 0])
        # aligned = backward_warp(gt_data[:, 0], flow)
        aligned = flow_warp(gt_data[:, 0], flow.permute(0, 2, 3, 1))

        pix_loss_val = self.pix_crit_val(
            aligned,
            gt_data[:, 1],
        )

        self.log_dict(
            {
                "val_pix_loss": pix_loss_val,
            },
            on_epoch=True,
            prog_bar=True,
        )

        return (
            (
                gt_data[:, 0],
                gt_data[:, 1],
                aligned,
            ),
            ("gt_pev vs gt_t vs aligned",),
        )
