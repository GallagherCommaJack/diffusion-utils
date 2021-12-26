from abc import abstractmethod
from copy import deepcopy
from itertools import chain

import kornia.augmentation as K

import torch
from torch import optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from diffusion_utils.utils import *

import wandb
from tqdm import tqdm, trange  # type: ignore

# TODO: sampling loop as separate method
# TODO: alternate sampling schedules


class BaseDiffusion(pl.LightningModule):
    def __init__(
        self,
        net,
        ema_init_steps=12000,
        ema_decay=0.999,
        lr=1e-4,
        do_gather=True,
        distill_weight=None,
    ):
        super().__init__()
        self.model = net
        self.model_ema = deepcopy(self.model)
        self.model_ema.eval().requires_grad_(False)
        self.ema_init_steps = ema_init_steps
        self.ema_decay = ema_decay
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.lr = lr
        self.do_gather = do_gather
        self.distill_weight = distill_weight

    def forward(self, x, time, **kwargs):
        if self.training:
            return self.model(x, time, **kwargs)
        else:
            return self.model_ema(x, time, **kwargs)

    @torch.no_grad()
    def sample(self, z, n, steps, eta, show_progress=True, **kwargs):
        """Draws samples from a model given starting noise."""
        ts = torch.ones([n], device=self.device)

        # Create the noise schedule
        t = torch.linspace(1, 0, steps + 1, device=self.device)[:-1]
        alphas, sigmas = get_alphas_sigmas(t)
        ts = ts * t

        # The sampling loop
        inner_iter = ts
        if show_progress:
            inner_iter = tqdm(inner_iter)
        iter = enumerate(inner_iter)
        for i, t in iter:
            # Get the model output (eps, the predicted noise)
            v = self.forward(z, t, **kwargs).float()

            # Predict the noise and the denoised image
            pred = z * alphas[i] - v * sigmas[i]
            eps = z * sigmas[i] + v * alphas[i]

            # If we are not on the last timestep, compute the noisy image for the
            # next timestep.
            if i < steps - 1:
                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = (
                    eta
                    * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt()
                    * (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
                )
                adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

                # Recombine the predicted noise and predicted denoised image in the
                # correct proportions for the next step
                z = pred * alphas[i + 1] + eps * adjusted_sigma

                # Add the correct amount of fresh noise
                if eta:
                    z += torch.randn_like(z) * ddim_sigma
                True

        # If we are on the last timestep, output the denoised image
        return pred

    def default_step(self, prefix, batch, batch_idx):
        loss, log_dict = self.eval_batch(batch, do_gather=self.do_gather)
        log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.default_step("train", batch, batch_idx)

    def on_before_zero_grad(self, *args, **kwargs):
        decay = (
            0.95 if self.trainer.global_step < self.ema_init_steps else self.ema_decay
        )
        ema_update(self.model, self.model_ema, decay)

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.lr)
        return opt

    def noise_reals(self, reals):
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        log_snrs = get_ddpm_schedule(t)
        alphas, sigmas = get_alphas_sigmas(log_snrs)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas
        return t, noised_reals, targets

    def prepare_stats(self, e, do_gather=False):
        l = e.mean()
        e = e.detach()

        if do_gather:
            e = self.all_gather(e)

        log_dict = {
            k: v
            for k, v in zip(
                ["loss", "variance", "skewness", "kurtosis"],
                calculate_stats(e),
            )
        }

        return l, log_dict

    def eval_reals(self, reals, do_gather=False, **kwargs):
        t, noised_reals, targets = self.noise_reals(reals)

        log_dict = {}
        loss = 0.0
        if self.distill_weight:
            t_out = torch.zeros_like(t)
            v, d_e = calc_v_with_distillation_errors(self, noised_reals, t, t_out)
            d_loss, d_logs = self.prepare_stats(d_e, do_gather=do_gather)
            log_dict.update({f"distillation_{k}": v for k, v in d_logs.items()})
            loss = loss + d_loss * self.distill_weight
        else:
            v = self.forward(noised_reals, t, **kwargs)
        e = v.sub(targets).pow(2)
        l, logs = self.prepare_stats(e)
        loss = loss + l
        log_dict.update({f"diffusion_{k}": v for k, v in logs.items()})
        log_dict["total"] = loss.item()
        return loss, log_dict


class UnconditionalDiffusion(BaseDiffusion):
    def eval_batch(self, batch, do_gather=False):
        reals = batch[0]
        return self.eval_reals(reals, do_gather)


class ClassConditionalDiffusion(BaseDiffusion):
    def eval_batch(self, batch, do_gather=False):
        reals, classes = batch[:2]
        return self.eval_reals(reals, classes=classes, do_gather=do_gather)


class VectorConditionalDiffusion(BaseDiffusion):
    def eval_batch(self, batch, do_gather=False):
        reals, cond = batch[:2]
        return self.eval_reals(reals, cond=cond)


class PatchSRDiffusion(BaseDiffusion):
    def __init__(self, factor, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.mk_crop = K.RandomCrop(patch_size)
        self.factor = factor

    def resample(self, x):
        down = F.interpolate(x, scale_factor=1 / self.factor)
        resampled = F.interpolate(down, scale_factor=self.factor)
        return resampled

    def eval_batch(self, batch, do_gather=False):
        reals = batch[0]
        reals = self.mk_crop(reals)
        resampled_reals = self.resample(reals)
        t, noised_reals, targets = self.noise_reals(reals)
        x = torch.cat([resampled_reals, noised_reals], dim=1)
        v = self.forward(x, time=t)
        return self.prepare_stats(v, targets, do_gather)

    def sample(self, imgs, n, steps, eta, show_progress=True):
        """Draws samples from a model given starting noise."""
        range_fn = trange if show_progress else range

        up = F.interpolate(imgs, scale_factor=self.factor)
        x = torch.randn_like(up)

        ts = torch.ones([n], device=self.device)

        # Create the noise schedule
        t = torch.linspace(1, 0, steps + 1, device=self.device)[:-1]
        alphas, sigmas = get_alphas_sigmas(t)

        # The sampling loop
        for i in range_fn(steps):
            # Get the model output (eps, the predicted noise)
            inp = torch.cat([up, x], dim=1)
            v = self.forward(inp, ts * t[i]).float()

            # Predict the noise and the denoised image
            pred = x * alphas[i] - v * sigmas[i]
            eps = x * sigmas[i] + v * alphas[i]

            # If we are not on the last timestep, compute the noisy image for the
            # next timestep.
            if i < steps - 1:
                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = (
                    eta
                    * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt()
                    * (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
                )
                adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

                # Recombine the predicted noise and predicted denoised image in the
                # correct proportions for the next step
                x = pred * alphas[i + 1] + eps * adjusted_sigma

                # Add the correct amount of fresh noise
                if eta:
                    x += torch.randn_like(x) * ddim_sigma

        # If we are on the last timestep, output the denoised image
        return pred


class WandbDemoCallback(pl.Callback):
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, module):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            grid = module.demo()

        if isinstance(grid, list):
            log_dict = {f"demo grid {k}": wandb.Image(v) for k, v in grid}
        else:
            log_dict = {"demo grid": wandb.Image(grid)}

        log_dict["global_step"] = trainer.global_step
        trainer.logger.experiment.log(log_dict)
