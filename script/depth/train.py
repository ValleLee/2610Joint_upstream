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

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import logging
import os
import shutil
import subprocess
import json
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime, timedelta
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from typing import List, Union

from marigold import MarigoldDepthPipeline
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_trainer_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.slurm_util import get_local_scratch_dir, is_on_slurm


def _make_head_tail_subset(dataset, head_n: int, tail_n: int):
    n = len(dataset)
    head_n = max(int(head_n), 0)
    tail_n = max(int(tail_n), 0)
    if n <= 0 or (head_n <= 0 and tail_n <= 0):
        return dataset
    indices = list(range(min(head_n, n)))
    if tail_n > 0:
        indices.extend(range(max(0, n - tail_n), n))
    indices = list(dict.fromkeys(int(i) for i in indices))
    if len(indices) >= n:
        return dataset
    return Subset(dataset, indices)


if "__main__" == __name__:
    t_start = datetime.now()
    logging.info(f"Started at {t_start}")

    distributed = False
    local_rank = 0
    rank = 0
    world_size = 1
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        distributed = True
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Monocular Depth Estimation : Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold_depth.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save checkpoints."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Run without Weights and Biases logging.",
    )
    parser.add_argument(
        "--do_not_copy_data",
        action="store_true",
        help="On Slurm cluster, do not copy data to the local scratch.",
    )
    parser.add_argument(
        "--base_data_dir", type=str, default=None, help="Base path to the datasets."
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="Base path to the pretrained checkpoints.",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name.",
    )
    parser.add_argument(
        "--no_datetime_prefix",
        action="store_true",
        help="Disable datetime prefix in the output folder name.",
    )
    parser.add_argument(
        "--skip_code_snapshot",
        action="store_true",
        help="Skip saving the code snapshot tarball.",
    )

    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ["BASE_CKPT_DIR"]
    )

    if distributed:
        if torch.cuda.is_available() and not args.no_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    # -------------------- Initialization --------------------
    # Resume previous run
    if resume_run is not None:
        logging.info(f"Resuming run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        # Run from start
        cfg = recursive_load_config(args.config)
        # Full job name
        pure_job_name = os.path.basename(args.config).split(".")[0]
        # Add time prefix (default: enabled)
        add_datetime_prefix = (not args.no_datetime_prefix) or args.add_datetime_prefix
        if add_datetime_prefix:
            job_name = f"{t_start.strftime('%Y%m%d_%H%M%S')}_{pure_job_name}"
        else:
            job_name = pure_job_name

        # Output dir
        out_dir_run = None
        if (not distributed) or rank == 0:
            if output_dir is not None:
                out_dir_run = os.path.join(output_dir, job_name)
            else:
                out_dir_run = os.path.join("./output", job_name)
            if os.path.exists(out_dir_run):
                base_out_dir_run = out_dir_run
                n = 1
                while os.path.exists(out_dir_run):
                    out_dir_run = f"{base_out_dir_run}_{n:02d}"
                    n += 1
            os.makedirs(out_dir_run, exist_ok=False)

        if distributed:
            obj_list = [out_dir_run]
            dist.broadcast_object_list(obj_list, src=0)
            out_dir_run = obj_list[0]
            dist.barrier()

    cfg_data = cfg.dataset

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    if (not distributed) or rank == 0:
        if not os.path.exists(out_dir_ckpt):
            os.makedirs(out_dir_ckpt)
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    if (not distributed) or rank == 0:
        if not os.path.exists(out_dir_tb):
            os.makedirs(out_dir_tb)
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    if (not distributed) or rank == 0:
        if not os.path.exists(out_dir_eval):
            os.makedirs(out_dir_eval)
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    if (not distributed) or rank == 0:
        if not os.path.exists(out_dir_vis):
            os.makedirs(out_dir_vis)

    if distributed:
        dist.barrier()

    # -------------------- Logging settings --------------------
    if (not distributed) or rank == 0:
        config_logging(cfg.logging, out_dir=out_dir_run)
    else:
        config_logging(cfg.logging, out_dir=None)
    logging.debug(f"config: {cfg}")

    # Initialize wandb
    if (not distributed) or rank == 0:
        enable_wandb = not args.no_wandb
    else:
        enable_wandb = False

    if enable_wandb:
        if resume_run is not None:
            wandb_id = load_wandb_job_id(out_dir_run)
            wandb_cfg_dict = {
                "id": wandb_id,
                "resume": "must",
                **cfg.wandb,
            }
        else:
            wandb_cfg_dict = {
                "config": dict(cfg),
                "name": job_name,
                "mode": "online",
                **cfg.wandb,
            }
        wandb_cfg_dict.update({"dir": out_dir_run})
        try:
            wandb_run = init_wandb(enable=True, **wandb_cfg_dict)
            save_wandb_job_id(wandb_run, out_dir_run)
        except Exception as e:
            logging.warning(
                f"W&B init failed, continuing with W&B disabled. Error: {type(e).__name__}: {e}"
            )
            init_wandb(enable=False)
    else:
        init_wandb(enable=False)

    # Tensorboard (should be initialized after wandb)
    if (not distributed) or rank == 0:
        tb_logger.set_dir(out_dir_tb)
    else:
        tb_logger.set_dir(os.path.join(out_dir_tb, f"rank{rank}"))

    log_slurm_job_id(step=0)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")
    if cuda_avail:
        try:
            logging.info(
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', None)}"
            )
            logging.info(f"torch.cuda.device_count()={torch.cuda.device_count()}")
            logging.info(f"torch.cuda.current_device()={torch.cuda.current_device()}")
            for _i in range(torch.cuda.device_count()):
                _name = torch.cuda.get_device_name(_i)
                try:
                    _free, _total = torch.cuda.mem_get_info(_i)
                except TypeError:
                    with torch.cuda.device(_i):
                        _free, _total = torch.cuda.mem_get_info()
                logging.info(
                    f"cuda:{_i} name={_name}, free={_free/1024**3:.2f}GiB, total={_total/1024**3:.2f}GiB"
                )
        except Exception as e:
            logging.warning(f"Failed to query CUDA device info: {type(e).__name__}: {e}")

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None and ((not distributed) or rank == 0):
        try:
            pairs_dir_rel = str(cfg.dataset.train.dir)
            pairs_dir_abs = os.path.join(base_data_dir, pairs_dir_rel)
            train_pairs_txt = os.path.join(pairs_dir_abs, str(cfg.dataset.train.filenames))
            val_pairs_txt = None
            if cfg.dataset.val and len(cfg.dataset.val) > 0:
                val_pairs_txt = os.path.join(
                    os.path.join(base_data_dir, str(cfg.dataset.val[0].dir)),
                    str(cfg.dataset.val[0].filenames),
                )
            report_json = os.path.join(pairs_dir_abs, "report.json")
            pairs_report = None
            if os.path.exists(report_json):
                try:
                    with open(report_json, "r", encoding="utf-8") as f:
                        pairs_report = json.load(f)
                except Exception:
                    pairs_report = None
            cfg.runtime = {
                "base_data_dir": str(base_data_dir),
                "base_ckpt_dir": str(base_ckpt_dir),
                "pairs_dir_env": os.environ.get("PAIRS_DIR", None),
                "pairs_dir_rel": pairs_dir_rel,
                "pairs_dir_abs": pairs_dir_abs,
                "train_pairs_txt": train_pairs_txt,
                "val_pairs_txt": val_pairs_txt,
                "pairs_report_json": report_json,
                "pairs_report_args": (pairs_report or {}).get("args", None) if isinstance(pairs_report, dict) else None,
                "pairs_scene_counts": (pairs_report or {}).get("scene_counts", None) if isinstance(pairs_report, dict) else None,
                "pairs_splits": (pairs_report or {}).get("splits", None) if isinstance(pairs_report, dict) else None,
            }
        except Exception:
            pass

        try:
            if cfg.get("runtime", None) is None:
                cfg.runtime = {}
            cfg.runtime["env"] = {
                k: os.environ.get(k, None)
                for k in [
                    "CUDA_VISIBLE_DEVICES",
                    "MAX_ITER",
                    "SAVE_PERIOD",
                    "VAL_PERIOD",
                    "VIS_PERIOD",
                    "BACKUP_PERIOD",
                    "WARMUP_STEPS",
                    "VAL_DENOISE_STEPS",
                    "RESIZE_H",
                    "RESIZE_W",
                    "DEFAULT_PROC_RES",
                    "PAIRS_DIR",
                ]
                if os.environ.get(k, None) is not None
            }
        except Exception:
            pass

        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            _cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            OmegaConf.save(config=_cfg_resolved, f=f)
        logging.info(f"Config saved to {_output_path}")
        if not args.skip_code_snapshot:
            _temp_code_dir = os.path.join(out_dir_run, "code_tar")
            _code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
            logging.info(f"Saving code snapshot from: {PROJECT_ROOT}")
            subprocess.run(
                f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'",
                shell=True,
                check=True,
                cwd=PROJECT_ROOT,
            )
            subprocess.run(
                f"tar -cf {_code_snapshot_path} {_temp_code_dir}",
                shell=True,
                check=True,
            )
            subprocess.run(
                f"rm -rf {_temp_code_dir}",
                shell=True,
                check=True,
            )
            logging.info(f"Code snapshot saved to: {_code_snapshot_path}")

    if distributed:
        dist.barrier()

    # -------------------- Copy data to local scratch (Slurm) --------------------
    if is_on_slurm() and (not args.do_not_copy_data):
        # local scratch dir
        original_data_dir = base_data_dir
        base_data_dir = os.path.join(get_local_scratch_dir(), "Marigold_data")
        # copy data
        required_data_list = find_value_in_omegaconf("dir", cfg_data)
        # if cfg_train.visualize.init_latent_path is not None:
        #     required_data_list.append(cfg_train.visualize.init_latent_path)
        required_data_list = list(set(required_data_list))
        logging.info(f"Required_data_list: {required_data_list}")
        for d in tqdm(required_data_list, desc="Copy data to local scratch"):
            ori_dir = os.path.join(original_data_dir, d)
            dst_dir = os.path.join(base_data_dir, d)
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            if os.path.isfile(ori_dir):
                shutil.copyfile(ori_dir, dst_dir)
            elif os.path.isdir(ori_dir):
                shutil.copytree(ori_dir, dst_dir)
        logging.info(f"Data copied to: {base_data_dir}")

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    denom = cfg.dataloader.max_train_batch_size * (world_size if distributed else 1)
    accumulation_steps = eff_bs / denom
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    logging.info(
        f"Effective batch size (global): {eff_bs}, world_size: {world_size}, per_device_max_train_batch_size: {cfg.dataloader.max_train_batch_size}, accumulation steps: {accumulation_steps}"
    )

    # -------------------- Data --------------------
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    depth_transform: DepthNormalizerBase = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )
    train_dataset: Union[BaseDepthDataset, List[BaseDepthDataset]] = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
    )
    logging.debug("Augmentation: ", cfg.augmentation)
    if "mixed" == cfg_data.train.name:
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        _dl_kwargs = {
            "num_workers": cfg.dataloader.num_workers,
            "pin_memory": cfg.dataloader.pin_memory,
        }
        if cfg.dataloader.num_workers and cfg.dataloader.num_workers > 0:
            _dl_kwargs["persistent_workers"] = cfg.dataloader.persistent_workers
            _dl_kwargs["prefetch_factor"] = getattr(cfg.dataloader, "prefetch_factor", 2)
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            **_dl_kwargs,
        )
    else:
        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        _dl_kwargs = {
            "num_workers": cfg.dataloader.num_workers,
            "pin_memory": cfg.dataloader.pin_memory,
        }
        if cfg.dataloader.num_workers and cfg.dataloader.num_workers > 0:
            _dl_kwargs["persistent_workers"] = cfg.dataloader.persistent_workers
            _dl_kwargs["prefetch_factor"] = getattr(cfg.dataloader, "prefetch_factor", 2)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            generator=loader_generator,
            **_dl_kwargs,
        )
    # Validation dataset
    val_loaders: List[DataLoader] = []
    for _val_dict in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dict,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
            depth_transform=depth_transform,
        )
        _dl_kwargs = {
            "num_workers": cfg.dataloader.num_workers,
            "pin_memory": cfg.dataloader.pin_memory,
        }
        if cfg.dataloader.num_workers and cfg.dataloader.num_workers > 0:
            _dl_kwargs["persistent_workers"] = cfg.dataloader.persistent_workers
            _dl_kwargs["prefetch_factor"] = getattr(cfg.dataloader, "prefetch_factor", 2)
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,
            shuffle=False,
            **_dl_kwargs,
        )
        val_loaders.append(_val_loader)

    # Visualization dataset
    vis_loaders: List[DataLoader] = []
    _vis_cfg = getattr(cfg, "visualization", None)
    _vis_head_samples = int(getattr(_vis_cfg, "head_samples", 0) or 0)
    _vis_tail_samples = int(getattr(_vis_cfg, "tail_samples", 0) or 0)
    for _vis_dict in cfg_data.vis:
        _vis_dataset = get_dataset(
            _vis_dict,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
            depth_transform=depth_transform,
        )
        _vis_dataset = _make_head_tail_subset(
            _vis_dataset,
            head_n=_vis_head_samples,
            tail_n=_vis_tail_samples,
        )
        _dl_kwargs = {
            "num_workers": cfg.dataloader.num_workers,
            "pin_memory": cfg.dataloader.pin_memory,
        }
        if cfg.dataloader.num_workers and cfg.dataloader.num_workers > 0:
            _dl_kwargs["persistent_workers"] = cfg.dataloader.persistent_workers
            _dl_kwargs["prefetch_factor"] = getattr(cfg.dataloader, "prefetch_factor", 2)
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=1,
            shuffle=False,
            **_dl_kwargs,
        )
        vis_loaders.append(_vis_loader)

    # -------------------- Model --------------------
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = MarigoldDepthPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path), **_pipeline_kwargs
    )

    # -------------------- Trainer --------------------
    # Exit time
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    trainer_cls = get_trainer_cls(cfg.trainer.name)
    logging.debug(f"Trainer: {trainer_cls}")
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        device=device,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=val_loaders,
        vis_dataloaders=vis_loaders,
        rank=rank,
        world_size=world_size,
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=True
        )

    # -------------------- Training & Evaluation Loop --------------------
    try:
        trainer.train(t_end=t_end)
    except Exception as e:
        logging.exception(e)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
