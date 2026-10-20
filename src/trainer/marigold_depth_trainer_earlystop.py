import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.util.depth_transform import get_depth_normalizer
from src.util.metric import MetricTracker

from .marigold_depth_trainer import MarigoldDepthTrainer


class MarigoldDepthTrainerEarlyStop(MarigoldDepthTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        es_cfg = getattr(self.cfg.trainer, "early_stopping", None)
        self.early_stop_enabled = bool(getattr(es_cfg, "enabled", False)) if es_cfg is not None else False
        self.early_stop_start_iter = int(getattr(es_cfg, "start_iter", 0) or 0) if es_cfg is not None else 0
        self.early_stop_patience = int(getattr(es_cfg, "patience", 0) or 0) if es_cfg is not None else 0
        self.early_stop_min_delta = float(getattr(es_cfg, "min_delta", 0.0) or 0.0) if es_cfg is not None else 0.0
        self.early_stop_goal = str(
            getattr(es_cfg, "goal", self.main_val_metric_goal) or self.main_val_metric_goal
        ).lower()
        self.early_stop_bad_count = 0
        self.early_stop_best_metric = None

    def _get_current_main_metric(self):
        if not self.eval_results:
            return None
        first_dataset = next(iter(self.eval_results.keys()))
        metric_dict = self.eval_results.get(first_dataset, {})
        if self.main_val_metric not in metric_dict:
            return None
        return float(metric_dict[self.main_val_metric])

    @staticmethod
    def _joint_latent_metric_values(predicted, expected):
        rgb_loss = F.mse_loss(predicted[:, :4], expected[:, :4].float())
        depth_loss = F.mse_loss(predicted[:, 4:], expected[:, 4:].float())
        return {
            "rgb_latent_mse": rgb_loss.item(),
            "depth_latent_mse": depth_loss.item(),
            "joint_latent_mse": ((rgb_loss + depth_loss) / 2).item(),
        }

    @torch.no_grad()
    def validate_single_dataset(self, data_loader, metric_tracker, save_to_dir=None):
        """Use joint latent MSE for validation and decoded images for visualization."""
        if save_to_dir is not None:
            image_metrics = MetricTracker(*(func.__name__ for func in self.metric_funcs))
            super().validate_single_dataset(
                data_loader, image_metrics, save_to_dir=save_to_dir
            )

        self.model.to(self.device)
        self.model.unet.eval()
        metric_tracker.reset()
        normalizer = get_depth_normalizer(self.cfg.depth_normalization)
        dataset = data_loader.dataset
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        for i, batch in enumerate(data_loader):
            rgb = batch["rgb_norm"].to(self.device)
            gt_rgb = batch["gt_rgb_norm"].to(self.device)
            condition = batch["condition_raw_linear"]
            condition_valid = batch["valid_mask_condition"].bool()
            teacher = batch["teacher_target_linear"]
            teacher_valid = batch["valid_mask_teacher_target"].bool()
            condition = normalizer(condition, condition_valid).clone()
            teacher = normalizer(teacher, teacher_valid).clone()
            if dataset.move_invalid_to_far_plane:
                condition[~condition_valid] = dataset._get_invalid_fill_value(
                    is_inverse=dataset.condition_is_inverse
                )
                teacher[~teacher_valid] = dataset._get_invalid_fill_value(
                    is_inverse=dataset.teacher_is_inverse
                )
            condition = condition.to(self.device)
            teacher = teacher.to(self.device)
            rgb_latent = self.encode_rgb(rgb)
            condition_latent = self.encode_depth(condition)
            targets = torch.cat(
                [self.encode_rgb(gt_rgb), self.encode_depth(teacher)], dim=1
            )
            generator = torch.Generator(device=self.device).manual_seed(
                int(self.cfg.validation.init_seed) + i
            )
            timestep = torch.full(
                (rgb.shape[0],),
                self.scheduler_timesteps // 2,
                device=self.device,
                dtype=torch.long,
            )
            noise = torch.randn(
                targets.shape,
                device=self.device,
                dtype=targets.dtype,
                generator=generator,
            )
            noisy = self.training_noise_scheduler.add_noise(
                targets, noise, timestep
            )
            model_input = torch.cat(
                [rgb_latent, condition_latent, noisy], dim=1
            ).float()
            embed = self.empty_text_embed.to(self.device).repeat(rgb.shape[0], 1, 1)
            predicted = self.model.unet(model_input, timestep, embed).sample.float()
            if self.prediction_type == "epsilon":
                expected = noise
            elif self.prediction_type == "sample":
                expected = targets
            elif self.prediction_type == "v_prediction":
                expected = self.training_noise_scheduler.get_velocity(
                    targets, noise, timestep
                )
            else:
                raise ValueError(self.prediction_type)
            for name, value in self._joint_latent_metric_values(
                predicted, expected
            ).items():
                metric_tracker.update(name, value)
        return metric_tracker.result()

    def _is_early_stop_improved(self, current_metric: float) -> bool:
        if self.early_stop_best_metric is None:
            return True
        if self.early_stop_goal == "maximize":
            return current_metric > (self.early_stop_best_metric + self.early_stop_min_delta)
        return current_metric < (self.early_stop_best_metric - self.early_stop_min_delta)

    def _broadcast_stop_flag(self, should_stop: bool) -> bool:
        if not self.distributed:
            return should_stop
        flag = torch.tensor(
            [1 if should_stop else 0],
            device=self.device,
            dtype=torch.int32,
        )
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def _train_step_callback(self):
        should_stop = False

        if self.is_main_process:
            if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
                self.save_checkpoint(
                    ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                )

            is_latest_saved = False
            active_val_period = self._get_active_period(
                early_period=self.early_val_period,
                late_period=self.val_period,
            )
            active_vis_period = self._get_active_period(
                early_period=self.early_vis_period,
                late_period=self.vis_period,
            )

            if self._should_run_period(active_val_period):
                self.in_evaluation = True
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                is_latest_saved = True
                self.validate()
                self.in_evaluation = False
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)

                if (
                    self.early_stop_enabled
                    and self.early_stop_patience > 0
                    and self.effective_iter >= self.early_stop_start_iter
                ):
                    current_metric = self._get_current_main_metric()
                    if current_metric is not None:
                        if self._is_early_stop_improved(current_metric):
                            self.early_stop_best_metric = current_metric
                            self.early_stop_bad_count = 0
                            logging.info(
                                f"[early-stop] reset patience at iter {self.effective_iter}: "
                                f"{self.main_val_metric}={current_metric:.6f}"
                            )
                        else:
                            self.early_stop_bad_count += 1
                            logging.info(
                                f"[early-stop] no sufficient improvement at iter {self.effective_iter}: "
                                f"{self.main_val_metric}={current_metric:.6f}, "
                                f"best={self.early_stop_best_metric:.6f}, "
                                f"bad_count={self.early_stop_bad_count}/{self.early_stop_patience}, "
                                f"min_delta={self.early_stop_min_delta}"
                            )
                            if self.early_stop_bad_count >= self.early_stop_patience:
                                should_stop = True
                                logging.info(
                                    f"[early-stop] triggered at iter {self.effective_iter}. "
                                    f"Stopping training after patience={self.early_stop_patience}."
                                )

            if (
                self.save_period > 0
                and 0 == self.effective_iter % self.save_period
                and not is_latest_saved
            ):
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)

            if self._should_run_period(active_vis_period):
                self.visualize()

        should_stop = self._broadcast_stop_flag(should_stop)
        if should_stop and (
            self.max_iter <= 0 or self.effective_iter < self.max_iter
        ):
            # Force the existing train-loop exit path to stop at the current iter.
            self.max_iter = self.effective_iter
