import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from PIL import Image

from src.trainer.marigold_depth_trainer import MarigoldDepthTrainer
from src.trainer.marigold_depth_trainer_earlystop import MarigoldDepthTrainerEarlyStop
from src.util.metric import MetricTracker, rmse_linear


class DummyUNet:
    def __init__(self):
        self.conv_in = nn.Conv2d(8, 2, kernel_size=3, padding=1, bias=True)
        self.conv_out = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=True)
        self.config = {"in_channels": 8, "out_channels": 4}

    def register_to_config(self, **kwargs):
        self.config.update(kwargs)


class JointTeacherLocalContractTest(unittest.TestCase):
    def setUp(self):
        self.trainer = MarigoldDepthTrainer.__new__(MarigoldDepthTrainer)
        self.unet = DummyUNet()
        self.trainer.model = SimpleNamespace(unet=self.unet)
        with torch.no_grad():
            self.unet.conv_in.weight.copy_(
                torch.arange(self.unet.conv_in.weight.numel(), dtype=torch.float32).reshape_as(
                    self.unet.conv_in.weight
                )
            )
            self.unet.conv_in.bias.copy_(torch.tensor([0.25, -0.5]))
            self.unet.conv_out.weight.copy_(
                torch.arange(self.unet.conv_out.weight.numel(), dtype=torch.float32).reshape_as(
                    self.unet.conv_out.weight
                )
            )
            self.unet.conv_out.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_conv_in_gives_both_noisy_targets_the_pretrained_noisy_weights(self):
        old_weight = self.unet.conv_in.weight.detach().clone()

        self.trainer._replace_unet_conv_in(in_channels=16)

        new_weight = self.unet.conv_in.weight.detach()
        torch.testing.assert_close(new_weight[:, 8:12], old_weight[:, 4:8])
        torch.testing.assert_close(new_weight[:, 12:16], old_weight[:, 4:8])

    def test_conv_out_gives_both_joint_targets_the_pretrained_head(self):
        old_weight = self.unet.conv_out.weight.detach().clone()
        old_bias = self.unet.conv_out.bias.detach().clone()

        self.trainer._replace_unet_conv_out(out_channels=8)

        torch.testing.assert_close(self.unet.conv_out.weight[:4], old_weight)
        torch.testing.assert_close(self.unet.conv_out.weight[4:], old_weight)
        torch.testing.assert_close(self.unet.conv_out.bias[:4], old_bias)
        torch.testing.assert_close(self.unet.conv_out.bias[4:], old_bias)
        self.assertEqual(self.unet.config["out_channels"], 8)

    def test_joint_training_loss_uses_every_rgb_and_depth_latent_element(self):
        self.assertTrue(
            hasattr(self.trainer, "_compute_joint_latent_loss"),
            "trainer must expose the full-tensor joint latent loss",
        )
        self.trainer.loss = nn.MSELoss(reduction="mean")
        predicted = torch.zeros((1, 8, 1, 2), dtype=torch.float32)
        expected = torch.tensor(
            [[[[1.0, 2.0]]] * 4 + [[[3.0, 4.0]]] * 4], dtype=torch.float32
        )

        actual = self.trainer._compute_joint_latent_loss(predicted, expected)

        torch.testing.assert_close(actual, torch.tensor(7.5))

    def test_joint_validation_reports_rgb_depth_and_equal_channel_joint_mse(self):
        validator = MarigoldDepthTrainerEarlyStop.__new__(MarigoldDepthTrainerEarlyStop)
        self.assertTrue(
            hasattr(validator, "_joint_latent_metric_values"),
            "validator must report joint RGB/depth latent metrics",
        )
        predicted = torch.zeros((1, 8, 1, 1), dtype=torch.float32)
        expected = torch.cat(
            [torch.ones((1, 4, 1, 1)), torch.full((1, 4, 1, 1), 2.0)], dim=1
        )

        values = validator._joint_latent_metric_values(predicted, expected)

        self.assertEqual(values, {
            "rgb_latent_mse": 1.0,
            "depth_latent_mse": 4.0,
            "joint_latent_mse": 2.5,
        })

    def test_visualization_saves_decoded_rgb_and_depth_without_changing_joint_metrics(self):
        class FakeUNet:
            conv_in = SimpleNamespace(in_channels=16)

            def eval(self):
                return self

            def __call__(self, latent, timestep, embed):
                return SimpleNamespace(sample=torch.zeros_like(latent[:, 8:16]))

        class FakePipeline:
            def __init__(self, depth):
                self.depth = depth
                self.unet = FakeUNet()
                self.vae = SimpleNamespace(eval=lambda: None)
                self.text_encoder = SimpleNamespace(eval=lambda: None)
                self.condition_seen = False

            def to(self, device):
                return self

            def __call__(self, rgb_int, **kwargs):
                self.condition_seen = kwargs["condition_norm"] is not None
                return SimpleNamespace(
                    depth_np=self.depth * 0.8 + 0.1,
                    rgb_np=np.full((8, 8, 3), 0.5, dtype=np.float32),
                )

        depth = np.linspace(1.0, 2.0, 64, dtype=np.float32).reshape(8, 8)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        valid = torch.ones_like(depth_tensor, dtype=torch.bool)
        batch = {
            "rgb_int": torch.zeros((1, 3, 8, 8), dtype=torch.uint8),
            "rgb_norm": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "gt_rgb_norm": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "condition_raw_linear": depth_tensor,
            "valid_mask_condition": valid,
            "teacher_target_linear": depth_tensor,
            "valid_mask_teacher_target": valid,
            "depth_raw_linear": depth_tensor,
            "valid_mask_raw": valid,
            "rgb_relative_path": ["scene/frame.png"],
        }

        class FakeLoader:
            batch_size = 1
            dataset = SimpleNamespace(
                disp_name="vis", resize_to_hw=None, teacher_is_inverse=False,
                move_invalid_to_far_plane=False, min_depth=0.0, max_depth=3.0,
            )

            def __iter__(self):
                yield batch

            def __len__(self):
                return 1

        trainer = MarigoldDepthTrainerEarlyStop.__new__(MarigoldDepthTrainerEarlyStop)
        trainer.device = torch.device("cpu")
        trainer.model = FakePipeline(depth)
        trainer.metric_funcs = [rmse_linear]
        trainer.val_log_per_sample = False
        trainer.encode_rgb = lambda x: torch.zeros((x.shape[0], 4, 1, 1))
        trainer.encode_depth = lambda x: torch.zeros((x.shape[0], 4, 1, 1))
        trainer.training_noise_scheduler = SimpleNamespace(
            add_noise=lambda targets, noise, timestep: targets + noise
        )
        trainer.scheduler_timesteps = 1000
        trainer.prediction_type = "epsilon"
        trainer.empty_text_embed = torch.zeros((1, 2, 3))
        trainer.cfg = SimpleNamespace(
            depth_normalization=SimpleNamespace(
                type="scale_shift_depth", norm_min=-1.0, norm_max=1.0,
                min_max_quantile=0.01, clip=True,
            ),
            validation=SimpleNamespace(
                init_seed=2024, match_input_res=False,
                target_representation="depth", denoising_steps=4,
                ensemble_size=1, processing_res=0, resample_method="bilinear",
                clip_to_dataset_range=False,
            ),
            eval=SimpleNamespace(alignment="least_square", align_max_res=None),
        )
        joint_tracker = MetricTracker("joint_latent_mse", "rgb_latent_mse", "depth_latent_mse")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = trainer.validate_single_dataset(
                FakeLoader(), joint_tracker, save_to_dir=temp_dir
            )
            stem = Path(temp_dir) / "scene_frame.png"
            rgb_path = Path(f"{stem}_rgb.png")
            depth_png_path = Path(f"{stem}.png")
            depth_npy_path = Path(f"{stem}.npy")
            self.assertTrue(rgb_path.is_file())
            self.assertTrue(depth_png_path.is_file())
            self.assertTrue(depth_npy_path.is_file())
            with Image.open(rgb_path) as rgb_image:
                self.assertEqual(rgb_image.size, (8, 8))
            with Image.open(depth_png_path) as depth_image:
                self.assertEqual(depth_image.size, (8, 8))
            self.assertEqual(np.load(depth_npy_path).shape, (8, 8))
            self.assertTrue(trainer.model.condition_seen)
            self.assertEqual(set(result), {
                "joint_latent_mse", "rgb_latent_mse", "depth_latent_mse",
            })
            self.assertGreater(result["joint_latent_mse"], 0)
            self.assertEqual(result, joint_tracker.result())

    def test_remote_scale_and_schedule_defaults_are_preserved(self):
        config_text = (
            Path(__file__).parents[1] / "config" / "train_rgbdepth_teacher_perimg.yaml"
        ).read_text(encoding="utf-8")
        for required in (
            "MAX_ITER,15000",
            "EFFECTIVE_BATCH_SIZE,64",
            "MAX_TRAIN_BATCH_SIZE,16",
            "LR_TOTAL_ITER,18000",
            "WARMUP_STEPS,200",
        ):
            self.assertIn(required, config_text)


if __name__ == "__main__":
    unittest.main()
