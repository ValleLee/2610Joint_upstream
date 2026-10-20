# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import logging
import os
import numpy as np
import random
import shutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None
from PIL import Image
from datetime import datetime
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Union

from marigold.marigold_depth_pipeline import MarigoldDepthPipeline, MarigoldDepthOutput
from src.util import metric
from src.util.alignment import align_depth_least_square
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence
from src.util.depth_transform import get_depth_normalizer


class MarigoldDepthTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldDepthPipeline,
        train_dataloader: DataLoader,
        device,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.cfg: OmegaConf = cfg
        self.model: MarigoldDepthPipeline = model
        self.device = device
        #分块加速不影响效果
        self.model.vae.enable_tiling()

        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.distributed = (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        self.is_main_process = (self.rank == 0)
        self.ddp_unet: Union[DistributedDataParallel, None] = None

        # Adapt input layers
        # Joint teacher target: [GT RGB latent, DA depth latent]. Keep the copied
        # YAML hyperparameters intact; the target adds four noisy latent channels.
        self.unet_in_channels = int(getattr(self.cfg.trainer, "unet_in_channels", 8)) + 4
        if self.unet_in_channels != int(self.model.unet.config["in_channels"]):
            self._replace_unet_conv_in(in_channels=self.unet_in_channels)
        if int(self.model.unet.config["out_channels"]) != 8:
            self._replace_unet_conv_out(out_channels=8)

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self.model.unet.enable_xformers_memory_efficient_attention()
        else:
            logging.info(
                "Skipping xformers memory efficient attention because CUDA is not available."
            )

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(
            self.model.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )

        logging.info(
            "DDPM training noise scheduler config is updated: "
            f"rescale_betas_zero_snr = {self.training_noise_scheduler.config.rescale_betas_zero_snr}, "
            f"timestep_spacing = {self.training_noise_scheduler.config.timestep_spacing}"
        )

        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Inference DDIM scheduler (used for validation)
        self.model.scheduler = DDIMScheduler.from_config(
            self.training_noise_scheduler.config,
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(
            "joint_latent_mse", "rgb_latent_mse", "depth_latent_mse"
        )
        self.eval_results = {}

        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal

        assert self.main_val_metric == "joint_latent_mse"

        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        _val_cfg = getattr(self.cfg, "validation", None)
        self.val_log_per_sample = bool(getattr(_val_cfg, "log_per_sample", False))
        self.val_log_main_metric_only = bool(
            getattr(_val_cfg, "log_main_metric_only", True)
        )

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period
        self.schedule_switch_iter = int(
            getattr(self.cfg.trainer, "schedule_switch_iter", 0) or 0
        )
        self.early_val_period = int(
            getattr(self.cfg.trainer, "early_validation_period", self.val_period) or 0
        )
        self.early_vis_period = int(
            getattr(
                self.cfg.trainer, "early_visualization_period", self.vis_period
            )
            or 0
        )

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def _replace_unet_conv_in(self, in_channels: int):
        old_in = int(self.model.unet.conv_in.in_channels)
        old_out = int(self.model.unet.conv_in.out_channels)

        _weight = self.model.unet.conv_in.weight.clone()  # [Cout, Cin, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [Cout]

        if in_channels == old_in:
            return

        if in_channels > old_in:
            # Safer expansion: keep original channels unchanged, initialize new channels to zero.
            # Special case for our conditioning order change:
            #   old input (8ch): [rgb(4), noisy(4)]
            #   new input (12ch): [rgb(4), invdepth(4), noisy(4)]
            # We therefore copy old rgb weights to the invdepth channels (4:8) and keep new noisy channels (8:12) copied from old noisy (4:8).
            new_weight = _weight.new_zeros((old_out, in_channels, 3, 3))
            if old_in == 8 and in_channels == 16:
                # [rendered RGB, rendered depth, noisy GT RGB, noisy teacher depth]
                new_weight[:, 0:4, :, :] = _weight[:, 0:4, :, :]
                new_weight[:, 4:8, :, :] = _weight[:, 0:4, :, :]
                new_weight[:, 8:12, :, :] = _weight[:, 4:8, :, :]
                new_weight[:, 12:16, :, :] = _weight[:, 4:8, :, :]
            elif old_in == 8 and in_channels == 12:
                new_weight[:, 0:4, :, :] = _weight[:, 0:4, :, :]
                new_weight[:, 4:8, :, :] = _weight[:, 0:4, :, :]
                new_weight[:, 8:12, :, :] = _weight[:, 4:8, :, :]
            else:
                new_weight[:, :old_in, :, :] = _weight
            _weight = new_weight
        else:
            _weight = _weight[:, :in_channels, :, :]

        _new_conv_in = Conv2d(
            in_channels, old_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        ).to(device=_weight.device, dtype=_weight.dtype)
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")

        self.model.unet.config["in_channels"] = int(in_channels)
        logging.info("Unet config is updated")
        return

    def _replace_unet_conv_out(self, out_channels: int):
        old = self.model.unet.conv_out
        if old.out_channels == out_channels:
            return
        if old.out_channels != 4 or out_channels != 8:
            raise ValueError(
                f"Expected UNet output expansion 4 -> 8, got "
                f"{old.out_channels} -> {out_channels}"
            )
        replacement = Conv2d(
            old.in_channels,
            out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
        ).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            replacement.weight[:4].copy_(old.weight)
            replacement.weight[4:8].copy_(old.weight)
            replacement.bias[:4].copy_(old.bias)
            replacement.bias[4:8].copy_(old.bias)
        self.model.unet.conv_out = replacement
        self.model.unet.register_to_config(out_channels=out_channels)
        logging.info("Unet conv_out expanded to 8 channels for joint RGB and depth prediction")

    def _compute_joint_latent_loss(self, model_pred, target):
        """Compute one equal-weight MSE over the complete RGB-depth latent."""
        return self.loss(model_pred.float(), target.float())

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.distributed and self.ddp_unet is None:
            if torch.cuda.is_available() and device.type == "cuda":
                device_idx = torch.cuda.current_device()
                self.ddp_unet = DistributedDataParallel(
                    self.model.unet,
                    device_ids=[device_idx],
                    output_device=device_idx,
                    broadcast_buffers=False,
                )
            else:
                self.ddp_unet = DistributedDataParallel(
                    self.model.unet,
                    broadcast_buffers=False,
                )

        if self.in_evaluation:
            if self.is_main_process:
                logging.info(
                    "Last evaluation was not finished, will do evaluation before continue training."
                )
                self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            if self.distributed:
                try:
                    sampler = getattr(self.train_loader, "sampler", None)
                    if sampler is not None and hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(self.epoch)
                except Exception:
                    pass

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data
                rgb = batch["rgb_norm"].to(device)
                depth_gt_for_latent = batch[self.gt_depth_type].to(device)
                gt_rgb_for_latent = batch["gt_rgb_norm"].to(device)
                condition_for_latent = None
                if "condition_norm" in batch:
                    condition_for_latent = batch["condition_norm"].to(device)
                elif "invdepth_norm" in batch:
                    condition_for_latent = batch["invdepth_norm"].to(device)

                if self.gt_mask_type is not None:
                    valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                    if valid_mask_for_latent.dtype is not torch.bool:
                        valid_mask_for_latent = valid_mask_for_latent > 0.5
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
                    depth_valid_down = valid_mask_down.repeat((1, 4, 1, 1))
                    valid_mask_down = torch.cat(
                        [torch.ones_like(depth_valid_down), depth_valid_down], dim=1
                    )

                batch_size = rgb.shape[0]

                with torch.no_grad():
                    # Encode image
                    rgb_latent = self.encode_rgb(rgb)  # [B, 4, h, w]
                    condition_latent = None
                    if condition_for_latent is not None:
                        condition_latent = self.encode_depth(condition_for_latent)
                    # Encode GT depth
                    gt_target_latent = self.encode_depth(
                        depth_gt_for_latent
                    )  # [B, 4, h, w]
                    gt_rgb_latent = self.encode_rgb(gt_rgb_for_latent)
                    gt_target_latent = torch.cat([gt_rgb_latent, gt_target_latent], dim=1)  # [B, 8, h, w]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        gt_target_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        gt_target_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 8, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    gt_target_latent, noise, timesteps
                )  # [B, 8, h, w]

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # Concat rgb and target latents
                if condition_latent is None:
                    cat_latents = torch.cat([rgb_latent, noisy_latents], dim=1)
                else:
                    cat_latents = torch.cat(
                        [rgb_latent, condition_latent, noisy_latents], dim=1
                    )
                cat_latents = cat_latents.float()

                # Predict the noise residual
                _unet = self.ddp_unet if self.ddp_unet is not None else self.model.unet
                model_pred = _unet(cat_latents, timesteps, text_embed).sample  # [B, 8, h, w]
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")
                    if self.gt_mask_type is not None:
                        logging.warning(
                            f"valid_mask_for_latent true={int(valid_mask_for_latent.sum().item())}/{int(valid_mask_for_latent.numel())}, "
                            f"valid_mask_down true={int(valid_mask_down.sum().item())}/{int(valid_mask_down.numel())}"
                        )

                # Get the target for loss depending on the prediction type
                if "sample" == self.prediction_type:
                    target = gt_target_latent
                elif "epsilon" == self.prediction_type:
                    target = noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        gt_target_latent, noise, timesteps
                    )
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")

                # Both four-channel targets participate over the complete latent.
                latent_loss = self._compute_joint_latent_loss(model_pred, target)

                loss = latent_loss.mean()

                report_loss = loss.detach()
                if self.distributed:
                    dist.all_reduce(report_loss, op=dist.ReduceOp.SUM)
                    report_loss = report_loss / float(self.world_size)
                self.train_metrics.update("loss", report_loss.item())

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    if self.is_main_process:
                        tb_logger.log_dict(
                            {
                                f"train/{k}": v
                                for k, v in self.train_metrics.result().items()
                            },
                            global_step=self.effective_iter,
                        )
                        tb_logger.writer.add_scalar(
                            "lr",
                            self.lr_scheduler.get_last_lr()[0],
                            global_step=self.effective_iter,
                        )
                        tb_logger.writer.add_scalar(
                            "n_batch_in_epoch",
                            self.n_batch_in_epoch,
                            global_step=self.effective_iter,
                        )
                        logging.info(
                            f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                        )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        if self.is_main_process:
                            logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        if self.is_main_process:
                            logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def encode_rgb(self, image_in):
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        latent = self.model.encode_rgb(image_in)
        return latent

    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1).repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        if not self.is_main_process:
            return
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        active_val_period = self._get_active_period(
            early_period=self.early_val_period,
            late_period=self.val_period,
        )
        active_vis_period = self._get_active_period(
            early_period=self.early_vis_period,
            late_period=self.vis_period,
        )
        # Validation
        if self._should_run_period(active_val_period):
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self._should_run_period(active_vis_period):
            self.visualize()

    def _get_active_period(self, *, early_period: int, late_period: int) -> int:
        if self.schedule_switch_iter > 0 and self.effective_iter <= self.schedule_switch_iter:
            return int(early_period)
        return int(late_period)

    def _should_run_period(self, period: int) -> bool:
        return int(period) > 0 and self.effective_iter > 0 and 0 == (
            self.effective_iter % int(period)
        )

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = getattr(val_loader.dataset, "disp_name", None)
            if val_dataset_name is None and hasattr(val_loader.dataset, "dataset"):
                val_dataset_name = getattr(val_loader.dataset.dataset, "disp_name", None)
            if val_dataset_name is None:
                val_dataset_name = val_loader.dataset.__class__.__name__
            sample_list_path = getattr(val_loader.dataset, "filename_ls_path", None)
            if sample_list_path is None and hasattr(val_loader.dataset, "dataset"):
                sample_list_path = getattr(
                    val_loader.dataset.dataset, "filename_ls_path", None
                )
            val_metric_dict = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            self.eval_results[val_dataset_name] = val_metric_dict
            if self.val_log_main_metric_only:
                _main_val = float(val_metric_dict[self.main_val_metric])
                logging.info(
                    f"Iter {self.effective_iter}. Val `{val_dataset_name}`: {self.main_val_metric}={_main_val:.6f}"
                )
            else:
                _metric_text = ", ".join(
                    [f"{k}={float(v):.6f}" for k, v in val_metric_dict.items()]
                )
                logging.info(
                    f"Iter {self.effective_iter}. Val `{val_dataset_name}`: {_metric_text}"
                )
            tb_logger.log_dict(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dict_to_text(
                val_metrics=val_metric_dict,
                dataset_name=val_dataset_name,
                sample_list_path=sample_list_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dict[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = getattr(val_loader.dataset, "disp_name", None)
            if vis_dataset_name is None and hasattr(val_loader.dataset, "dataset"):
                vis_dataset_name = getattr(val_loader.dataset.dataset, "disp_name", None)
            if vis_dataset_name is None:
                vis_dataset_name = val_loader.dataset.__class__.__name__
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    def _convert_eval_representation_np(
        self,
        arr: np.ndarray,
        *,
        src_is_inverse: bool,
        dst_is_inverse: bool,
    ) -> np.ndarray:
        arr = arr.astype(np.float32, copy=False)
        if bool(src_is_inverse) == bool(dst_is_inverse):
            return arr
        valid = np.isfinite(arr) & (arr > 0)
        out = np.zeros_like(arr, dtype=np.float32)
        out[valid] = 1.0 / (arr[valid] + 1e-6)
        return out

    def _center_crop_eval_tensors(
        self,
        pred_arr: np.ndarray,
        gt_arr: np.ndarray,
        mask_arr: np.ndarray,
        gt_ts: torch.Tensor,
        mask_ts: torch.Tensor,
    ):
        pred_h, pred_w = pred_arr.shape[-2:]
        gt_h, gt_w = gt_arr.shape[-2:]
        mask_h, mask_w = mask_arr.shape[-2:]
        out_h = min(pred_h, gt_h, mask_h)
        out_w = min(pred_w, gt_w, mask_w)

        def _crop_np(x: np.ndarray) -> np.ndarray:
            h, w = x.shape[-2:]
            top = max((h - out_h) // 2, 0)
            left = max((w - out_w) // 2, 0)
            return x[top : top + out_h, left : left + out_w]

        def _crop_ts(x: torch.Tensor) -> torch.Tensor:
            h, w = x.shape[-2:]
            top = max((h - out_h) // 2, 0)
            left = max((w - out_w) // 2, 0)
            return x[..., top : top + out_h, left : left + out_w]

        return (
            _crop_np(pred_arr),
            _crop_np(gt_arr),
            _crop_np(mask_arr),
            _crop_ts(gt_ts),
            _crop_ts(mask_ts),
            out_h,
            out_w,
        )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        self.model.unet.eval()
        try:
            self.model.vae.eval()
        except Exception:
            pass
        try:
            self.model.text_encoder.eval()
        except Exception:
            pass
        metric_tracker.reset()

        warned_shape_mismatch = False
        warned_match_input_override = False

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        dataset_name = getattr(data_loader.dataset, "disp_name", None)
        if dataset_name is None and hasattr(data_loader.dataset, "dataset"):
            dataset_name = getattr(data_loader.dataset.dataset, "disp_name", None)
        if dataset_name is None:
            dataset_name = data_loader.dataset.__class__.__name__

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {dataset_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            _dataset_ref = data_loader.dataset
            if (
                not hasattr(_dataset_ref, "min_depth")
                and hasattr(_dataset_ref, "dataset")
            ):
                _dataset_ref = _dataset_ref.dataset
            # Read input image
            rgb_int = batch["rgb_int"]  # [B, 3, H, W]

            condition_norm = None
            if "condition_raw_linear" in batch or "invdepth_raw_linear" in batch:
                condition_ts = batch.get("condition_raw_linear", batch.get("invdepth_raw_linear"))
                condition_valid_ts = batch.get("valid_mask_condition", batch.get("valid_mask_invdepth", None))
                if condition_valid_ts is None:
                    condition_valid_ts = torch.isfinite(condition_ts) & (condition_ts > 0)
                depth_transform = get_depth_normalizer(self.cfg.depth_normalization)
                condition_norm = depth_transform(condition_ts, condition_valid_ts).clone()
                if getattr(data_loader.dataset, "move_invalid_to_far_plane", False):
                    if hasattr(_dataset_ref, "_get_invalid_fill_value") and (
                        getattr(_dataset_ref, "depth_transform", None) is not None
                    ):
                        condition_fill = _dataset_ref._get_invalid_fill_value(
                            is_inverse=bool(getattr(_dataset_ref, "condition_is_inverse", False))
                        )
                    else:
                        if getattr(depth_transform, "far_plane_at_max", False):
                            condition_fill = (
                                depth_transform.norm_min
                                if bool(getattr(_dataset_ref, "condition_is_inverse", False))
                                else depth_transform.norm_max
                            )
                        else:
                            condition_fill = (
                                depth_transform.norm_max
                                if bool(getattr(_dataset_ref, "condition_is_inverse", False))
                                else depth_transform.norm_min
                            )
                    condition_norm[~condition_valid_ts] = condition_fill
                if self.val_log_per_sample:
                    logging.debug(
                        f"[DBG condition_norm] min,mean,max={condition_norm.min().item():.6f},{condition_norm.mean().item():.6f},{condition_norm.max().item():.6f}; "
                        f"valid_ratio={condition_valid_ts.float().mean().item():.3f}"
                    )
            expected_in_channels = int(self.model.unet.conv_in.in_channels)
            if self.val_log_per_sample:
                logging.debug(f"[DBG pipeline] unet_in_channels={expected_in_channels}")
            eval_target_type = str(
                getattr(self.cfg.validation, "target_representation", "dataset")
            ).lower()
            teacher_is_inverse = bool(
                getattr(
                    _dataset_ref,
                    "teacher_is_inverse",
                    getattr(_dataset_ref, "teacher_from_depth", False),
                )
            )
            if eval_target_type == "dataset":
                target_use_inverse = teacher_is_inverse
            elif eval_target_type == "invdepth":
                target_use_inverse = True
            elif eval_target_type == "depth":
                target_use_inverse = False
            else:
                raise RuntimeError(
                    f"Unknown validation.target_representation={eval_target_type}; expected one of ['dataset', 'depth', 'invdepth']"
                )

            teacher_target_ts = batch.get("teacher_target_linear", batch["depth_raw_linear"]).squeeze()
            valid_mask_ts = batch.get("valid_mask_teacher_target", batch["valid_mask_raw"]).squeeze()
            teacher_target_native = teacher_target_ts.numpy().astype(np.float32, copy=False)
            valid_mask = valid_mask_ts.numpy()

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict depth
            _match_input_res = self.cfg.validation.match_input_res
            if (
                (not _match_input_res)
                and hasattr(data_loader.dataset, "resize_to_hw")
                and getattr(data_loader.dataset, "resize_to_hw") is not None
            ):
                _match_input_res = True
                if not warned_match_input_override:
                    logging.info(
                        "[validate_single_dataset] dataset.resize_to_hw is set; overriding match_input_res=True to keep pred/gt on the same grid."
                    )
                    warned_match_input_override = True
            pipe_out: MarigoldDepthOutput = self.model(
                rgb_int,
                condition_norm=condition_norm,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=_match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                color_map=None,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            depth_pred: np.ndarray = np.asarray(pipe_out.depth_np, dtype=np.float32)

            if depth_pred.shape != teacher_target_native.shape:
                _pred_shape = tuple(depth_pred.shape)
                _gt_shape = tuple(teacher_target_native.shape)
                _mask_shape = tuple(valid_mask.shape)
                depth_pred, teacher_target_native, valid_mask, teacher_target_ts, valid_mask_ts, _h, _w = self._center_crop_eval_tensors(
                    depth_pred, teacher_target_native, valid_mask, teacher_target_ts, valid_mask_ts
                )
                if (not warned_shape_mismatch) or self.val_log_per_sample:
                    _rgb_p = batch.get("rgb_relative_path", None)
                    _d_p = batch.get("depth_relative_path", None)
                    _t_p = batch.get("teacher_relative_path", None)
                    _inv_p = batch.get("invdepth_relative_path", None)
                    logging.warning(
                        f"[validate_single_dataset] pred/gt/mask shape mismatch, will center-crop to ({_h}, {_w}). "
                        f"pred={_pred_shape}, gt={_gt_shape}, mask={_mask_shape}. "
                        f"paths: rgb={_rgb_p}, depth={_d_p}, teacher={_t_p}, inv={_inv_p}"
                    )
                    warned_shape_mismatch = True

            # === [新增 调试 A：对齐前的量纲/掩膜] ===
            if self.val_log_per_sample:
                logging.debug(
                    f"[DBG pre-align] pred[min,mean,max]={depth_pred.min():.6f},{depth_pred.mean():.6f},{depth_pred.max():.6f}"
                )
                logging.debug(
                    f"[DBG pre-align]   gt_native[min,mean,max]={teacher_target_native.min():.6f},{teacher_target_native.mean():.6f},{teacher_target_native.max():.6f}"
                )
                logging.debug(
                    f"[DBG pre-align] valid_ratio={float(valid_mask.mean()):.3f}"
                )
            # ======================================

            if "least_square" == self.cfg.eval.alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=teacher_target_native,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=self.cfg.eval.align_max_res,
                )
                
                # === [新增 调试 B：对齐参数与对齐后的量纲] ===
                _scale = float(np.asarray(scale).item())
                _shift = float(np.asarray(shift).item())
                if self.val_log_per_sample:
                    logging.debug(
                        f"[DBG aligned] scale={_scale:.6f}, shift={_shift:.6f}, "
                        f"pred[min,mean,max]={depth_pred.min():.6f},{depth_pred.mean():.6f},{depth_pred.max():.6f}"
                    )
                # ==============================================
            else:
                raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")

            if target_use_inverse != teacher_is_inverse:
                depth_raw = self._convert_eval_representation_np(
                    teacher_target_native,
                    src_is_inverse=teacher_is_inverse,
                    dst_is_inverse=target_use_inverse,
                )
                depth_pred = self._convert_eval_representation_np(
                    depth_pred,
                    src_is_inverse=teacher_is_inverse,
                    dst_is_inverse=target_use_inverse,
                )
            else:
                depth_raw = teacher_target_native

            depth_raw_ts = torch.from_numpy(depth_raw.copy()).to(self.device)
            valid_mask_ts = valid_mask_ts.to(self.device)

            # Clip to dataset min max
            _eval_dmin = getattr(_dataset_ref, "min_depth", None)
            _eval_dmax = getattr(_dataset_ref, "max_depth", None)
            if target_use_inverse != teacher_is_inverse:
                if (_eval_dmin is not None) and (_eval_dmax is not None) and (_eval_dmin > 0) and (_eval_dmax > 0):
                    _a = 1.0 / (float(_eval_dmax) + 1e-6)
                    _b = 1.0 / (float(_eval_dmin) + 1e-6)
                    _eval_dmin, _eval_dmax = (min(_a, _b), max(_a, _b))
                else:
                    _eval_dmin, _eval_dmax = (None, None)

            _clip_eval_range = bool(
                getattr(self.cfg.validation, "clip_to_dataset_range", False)
            )
            if _clip_eval_range and (_eval_dmin is not None) and (_eval_dmax is not None):
                depth_pred = np.clip(depth_pred, a_min=_eval_dmin, a_max=_eval_dmax)
                _vis_dmin, _vis_dmax = _eval_dmin, _eval_dmax
            else:
                _vis_dmin, _vis_dmax = (None, None)

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)
            '''
            原来只存png改成下面都存npy和png
            但其实npy才是评测用的，png只是可视化
            # Save as 16-bit uint png
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                depth_to_save = (pipe_out.depth_np * 65535.0).astype(np.uint16)
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")
                # Save depth as both npy (float32, meters) and 16-bit PNG (normalized [0,1])
          
                '''  
            # Save depth as both npy (float32, meters) and 16-bit PNG (normalized [0,1])
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                rgb_save_path = os.path.join(save_to_dir, f"{img_name}_rgb.png")
                Image.fromarray((pipe_out.rgb_np * 255.0).round().astype(np.uint8)).save(rgb_save_path)

                # 保存 npy （保持浮点米制深度，便于后续 eval）
                npy_save_path = os.path.join(save_to_dir, f"{img_name}.npy")
                np.save(npy_save_path, depth_pred.astype(np.float32))

                # 保存 png （归一化到 [0,1] 再存 16-bit，可视化用）
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                dmin, dmax = _vis_dmin, _vis_dmax
                _m = np.isfinite(depth_pred) & (depth_pred > 0)
                _vals = depth_pred[_m] if _m.any() else None
                _vis_pcts = getattr(self.cfg.validation, "vis_percentiles", None)
                if _vals is not None:
                    if _vis_pcts is not None:
                        try:
                            p_lo, p_hi = float(_vis_pcts[0]), float(_vis_pcts[1])
                        except Exception:
                            p_lo, p_hi = None, None
                    else:
                        p_lo, p_hi = 1.0, 99.0
                    if (p_lo is not None) and (p_hi is not None):
                        _q_lo, _q_hi = np.percentile(_vals, [p_lo, p_hi])
                        if _q_hi - _q_lo > 1e-12:
                            dmin, dmax = float(_q_lo), float(_q_hi)
                    if (dmin is None) or (dmax is None) or (dmax - dmin <= 1e-12):
                        _vmin, _vmax = float(np.min(_vals)), float(np.max(_vals))
                        if _vmax - _vmin > 1e-12:
                            dmin, dmax = _vmin, _vmax
                if (dmin is None) or (dmax is None) or (dmax - dmin <= 1e-12):
                    depth_norm = np.zeros_like(depth_pred, dtype=np.float32)
                else:
                    depth_norm = np.clip((depth_pred - dmin) / (dmax - dmin), 0.0, 1.0)
                depth_to_save = (depth_norm * 65535.0).astype(np.uint16)
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")


        return metric_tracker.result()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        if not self.is_main_process:
            return
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=True)
        logging.info(f"UNet is saved to: {unet_path}")

        # Save scheduler
        scheduelr_path = os.path.join(ckpt_dir, "scheduler")
        self.model.scheduler.save_pretrained(scheduelr_path)
        logging.info(f"Scheduler is saved to: {scheduelr_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        unet_dir = os.path.join(ckpt_path, "unet")
        _unet_param = next(self.model.unet.parameters(), None)
        _target_device = (
            _unet_param.device if _unet_param is not None else torch.device(self.device)
        )
        _target_dtype = _unet_param.dtype if _unet_param is not None else None
        try:
            _load_kwargs = {}
            if _target_dtype is not None:
                _load_kwargs["torch_dtype"] = _target_dtype
            self.model.unet = self.model.unet.from_pretrained(unet_dir, **_load_kwargs)
        except Exception:
            _bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")
            _safetensors_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
            state_dict = None
            if os.path.exists(_safetensors_path):
                if safetensors_load_file is None:
                    raise ModuleNotFoundError(
                        "Found diffusion_pytorch_model.safetensors but safetensors is not available in this environment. "
                        "Please install safetensors or save checkpoints as .bin."
                    )
                state_dict = safetensors_load_file(
                    _safetensors_path, device=str(self.device)
                )
            elif os.path.exists(_bin_path):
                state_dict = torch.load(_bin_path, map_location=self.device)
            else:
                raise FileNotFoundError(
                    f"No diffusion_pytorch_model.safetensors or diffusion_pytorch_model.bin found under {unet_dir}"
                )
            self.model.unet.load_state_dict(state_dict)

        if _target_dtype is not None:
            self.model.unet = self.model.unet.to(
                device=_target_device, dtype=_target_dtype
            )
        else:
            self.model.unet = self.model.unet.to(device=_target_device)

        self.model.unet.requires_grad_(True)
        try:
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                self.model.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        lr = self.cfg.lr
        self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(
                os.path.join(ckpt_path, "trainer.ckpt"), map_location=self.device
            )
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
