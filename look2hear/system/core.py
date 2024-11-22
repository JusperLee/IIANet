###
# Author: Kai Li
# Date: 2021-06-19 11:43:37
# LastEditors: Kai Li
# LastEditTime: 2021-06-29 02:03:23
###

import torch
from pprint import pprint
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
import warnings

warnings.filterwarnings("ignore")


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class System(pl.LightningModule):
    default_monitor: str = "val_loss"

    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Save lightning"s AttributeDict under self.hparams
        self.save_hyperparameters(self.config_to_hparams(self.config))

    def forward(self, wav, mouth=None):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        if self.video_model == None:
            return self.audio_model(wav)
        else:
            with torch.no_grad():
                mouth_emb = self.video_model(mouth.type_as(wav))
            return self.audio_model(wav, mouth_emb)

    def common_step(self, batch, batch_nb):
        if self.video_model == None:
            if self.config["training"]["online_mix"] == True:
                inputs, targets, _ = self.online_mixing_collate(batch)
            else:
                inputs, targets, _ = batch
            est_targets = self(inputs)
            loss = self.loss_func(est_targets, targets)
            return loss
        elif self.video_model != None:
            inputs, targets, target_mouths, _ = batch
            est_targets = self(inputs, target_mouths)
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            loss = self.loss_func(est_targets, targets).mean()
            return loss

    def training_step(self, batch, batch_nb):
        loss = self.common_step(batch, batch_nb)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_loss = torch.mean(self.all_gather(avg_loss))
        # import pdb; pdb.set_trace()
        self.logger.experiment.log_metric(
            "train_sisnr", -train_loss, step=self.current_epoch
        )

    def validation_step(self, batch, batch_nb):
        loss = self.common_step(batch, batch_nb)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log_metric(
            "learning_rate",
            self.optimizer.param_groups[0]["lr"],
            step=self.current_epoch,
        )
        self.logger.experiment.log_metric(
            "val_sisnr", -val_loss, step=self.current_epoch
        )

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def online_mixing_collate(batch):
        """Mix target sources to create new mixtures.
        Output of the default collate function is expected to return two objects:
        inputs and targets.
        """
        # Inputs (batch, time) / targets (batch, n_src, time)
        inputs, targets = batch
        batch, n_src, _ = targets.shape

        energies = torch.sum(targets ** 2, dim=-1, keepdim=True)
        new_src = []
        for i in range(targets.shape[1]):
            new_s = targets[torch.randperm(batch), i, :]
            new_s = new_s * torch.sqrt(
                energies[:, i] / (new_s ** 2).sum(-1, keepdims=True)
            )
            new_src.append(new_s)

        targets = torch.stack(new_src, dim=1)
        inputs = targets.sum(1)
        return inputs, targets

    def on_epoch_end(self):
        if (
            self.config["sche"]["patience"] > 0
            and self.config["training"]["divide_lr_by"] != None
        ):
            if (
                self.current_epoch % self.config["sche"]["patience"] == 0
                and self.current_epoch != 0
            ):
                new_lr = self.config["optim"]["lr"] / (
                    self.config["training"]["divide_lr_by"]
                    ** (self.current_epoch // self.config["sche"]["patience"])
                )
                # print("Reducing Learning rate to: {}".format(new_lr))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
