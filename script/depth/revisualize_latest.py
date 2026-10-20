import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
from datetime import datetime

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from marigold import MarigoldDepthPipeline
from src.dataset import DatasetMode, get_dataset
from src.trainer import get_trainer_cls


def _resolve_base_dirs(cfg, args):
    base_data_dir = args.base_data_dir
    if base_data_dir is None:
        base_data_dir = getattr(getattr(cfg, "runtime", {}), "base_data_dir", None)
    if base_data_dir is None:
        base_data_dir = os.environ.get("BASE_DATA_DIR", None)
    if base_data_dir is None:
        raise ValueError("base_data_dir is not set. Provide --base_data_dir or set env BASE_DATA_DIR.")

    base_ckpt_dir = args.base_ckpt_dir
    if base_ckpt_dir is None:
        base_ckpt_dir = getattr(getattr(cfg, "runtime", {}), "base_ckpt_dir", None)
    if base_ckpt_dir is None:
        base_ckpt_dir = os.environ.get("BASE_CKPT_DIR", None)
    if base_ckpt_dir is None:
        raise ValueError("base_ckpt_dir is not set. Provide --base_ckpt_dir or set env BASE_CKPT_DIR.")

    return base_data_dir, base_ckpt_dir


def _dataset_name(ds):
    if hasattr(ds, "dataset") and hasattr(ds.dataset, "disp_name"):
        return ds.dataset.disp_name
    return ds.disp_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="latest")
    parser.add_argument("--split", type=str, default="vis", choices=["vis", "val"])
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--base_data_dir", type=str, default=None)
    parser.add_argument("--base_ckpt_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vis_dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    base_data_dir, base_ckpt_dir = _resolve_base_dirs(cfg, args)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model_dtype = dtype_map[str(args.dtype)]
    if device.type == "cpu" and model_dtype != torch.float32:
        model_dtype = torch.float32

    split_cfg = cfg.dataset.vis if args.split == "vis" else cfg.dataset.val
    loaders = []
    for ds_cfg in split_cfg:
        ds = get_dataset(ds_cfg, base_data_dir=base_data_dir, mode=DatasetMode.EVAL)
        if args.max_samples is not None and int(args.max_samples) > 0:
            ds = Subset(ds, list(range(min(int(args.max_samples), len(ds)))))
        loaders.append(
            DataLoader(
                dataset=ds,
                batch_size=1,
                shuffle=False,
                num_workers=int(cfg.dataloader.num_workers),
                pin_memory=bool(cfg.dataloader.pin_memory),
            )
        )

    pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = MarigoldDepthPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path),
        torch_dtype=model_dtype,
        **pipeline_kwargs,
    )

    trainer_cls = get_trainer_cls(cfg.trainer.name)
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=loaders[0],
        device=device,
        out_dir_ckpt=os.path.join(run_dir, "checkpoint"),
        out_dir_eval=os.path.join(run_dir, "evaluation"),
        out_dir_vis=os.path.join(run_dir, "visualization"),
        accumulation_steps=1,
        val_dataloaders=loaders,
        vis_dataloaders=[],
        rank=0,
        world_size=1,
    )

    ckpt_dir = os.path.join(run_dir, "checkpoint", args.ckpt_name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    trainer.load_checkpoint(ckpt_dir, load_trainer_state=False, resume_lr_scheduler=False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.vis_dir is not None:
        vis_root = os.path.abspath(args.vis_dir)
    else:
        vis_root = os.path.join(
            run_dir,
            "visualization",
            f"revisualize_{args.split}_{args.ckpt_name}_{stamp}",
        )
    os.makedirs(vis_root, exist_ok=True)

    results = {}
    for loader in loaders:
        name = _dataset_name(loader.dataset)
        save_to_dir = os.path.join(vis_root, name)
        os.makedirs(save_to_dir, exist_ok=True)
        metrics = trainer.validate_single_dataset(
            loader,
            trainer.val_metrics,
            save_to_dir=save_to_dir,
        )
        results[name] = {k: float(v) for k, v in metrics.items()}

    manifest = {
        "run_dir": run_dir,
        "split": args.split,
        "ckpt_name": args.ckpt_name,
        "max_samples": int(args.max_samples),
        "dtype": str(args.dtype),
        "device": str(device),
        "vis_root": vis_root,
        "results": results,
    }
    manifest_path = os.path.join(vis_root, "summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("saved", manifest_path)


if __name__ == "__main__":
    main()
