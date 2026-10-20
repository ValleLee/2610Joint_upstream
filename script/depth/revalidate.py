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
from src.util.depth_transform import get_depth_normalizer


def _get_default(val, default):
    return default if val is None else val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to an output run dir that contains config.yaml and checkpoint/",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="latest",
        help="Checkpoint folder name under run_dir/checkpoint (e.g. latest, iter_005000, iter_005005)",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=None,
        help="Override validation.denoising_steps; if omitted, follow config",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=None,
        help="Override validation.ensemble_size; if omitted, follow config",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Override validation.processing_res; use 0 to disable processing_res in pipeline",
    )
    parser.add_argument(
        "--match_input_res",
        type=int,
        default=None,
        help="Override validation.match_input_res (0/1). If omitted, follow config (and trainer may override to True when resize_to_hw is set)",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=None,
        help="Override BASE_DATA_DIR used to resolve dataset dirs. If omitted, use cfg.runtime.base_data_dir or env BASE_DATA_DIR",
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="Override BASE_CKPT_DIR used to resolve pretrained_path. If omitted, use cfg.runtime.base_ckpt_dir or env BASE_CKPT_DIR",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="If >0, evaluate only the first N samples of each val dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device string, e.g. cuda, cuda:0, cpu. Default: cuda if available else cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype for inference to reduce VRAM usage. fp16/bf16 are recommended on A100 GPUs.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save prediction png/npy using trainer.validate_single_dataset(save_to_dir=...).",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="Custom visualization output dir. Default: <run_dir>/visualization/revalidate_<tag>_<timestamp>",
    )

    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)

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

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    model_dtype = dtype_map[str(args.dtype)]
    if device.type == "cpu" and model_dtype != torch.float32:
        model_dtype = torch.float32

    if args.denoising_steps is not None:
        cfg.validation.denoising_steps = int(args.denoising_steps)
    if args.ensemble_size is not None:
        cfg.validation.ensemble_size = int(args.ensemble_size)
    if args.processing_res is not None:
        cfg.validation.processing_res = int(args.processing_res)
    if args.match_input_res is not None:
        cfg.validation.match_input_res = bool(int(args.match_input_res))

    depth_transform = get_depth_normalizer(cfg.depth_normalization)

    val_loaders = []
    for _val_dict in cfg.dataset.val:
        ds = get_dataset(
            _val_dict,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
            depth_transform=depth_transform,
        )
        if args.max_samples is not None and int(args.max_samples) > 0:
            n = min(int(args.max_samples), len(ds))
            ds = Subset(ds, list(range(n)))
        val_loaders.append(
            DataLoader(
                dataset=ds,
                batch_size=1,
                shuffle=False,
                num_workers=int(cfg.dataloader.num_workers),
                pin_memory=bool(cfg.dataloader.pin_memory),
            )
        )

    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = MarigoldDepthPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path),
        torch_dtype=model_dtype,
        **_pipeline_kwargs,
    )

    trainer_cls = get_trainer_cls(cfg.trainer.name)
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=val_loaders[0],
        device=device,
        out_dir_ckpt=os.path.join(run_dir, "checkpoint"),
        out_dir_eval=os.path.join(run_dir, "evaluation"),
        out_dir_vis=os.path.join(run_dir, "visualization"),
        accumulation_steps=1,
        val_dataloaders=val_loaders,
        vis_dataloaders=[],
        rank=0,
        world_size=1,
    )

    ckpt_dir = os.path.join(run_dir, "checkpoint", args.ckpt_name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    trainer.load_checkpoint(ckpt_dir, load_trainer_state=False, resume_lr_scheduler=False)

    results = {}
    tag_steps = int(cfg.validation.denoising_steps)
    tag = f"{args.ckpt_name}_steps{tag_steps}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_root = None
    if args.save_vis:
        if args.vis_dir is not None:
            vis_root = os.path.abspath(args.vis_dir)
        else:
            vis_root = os.path.join(
                run_dir, "visualization", f"revalidate_{tag}_{stamp}"
            )
        os.makedirs(vis_root, exist_ok=True)
    for loader in val_loaders:
        name = loader.dataset.dataset.disp_name if hasattr(loader.dataset, "dataset") else loader.dataset.disp_name
        save_to_dir = None
        if vis_root is not None:
            save_to_dir = os.path.join(vis_root, name)
            os.makedirs(save_to_dir, exist_ok=True)
        metrics = trainer.validate_single_dataset(
            loader, trainer.val_metrics, save_to_dir=save_to_dir
        )
        results[name] = {k: float(v) for k, v in metrics.items()}

    loss_dir = os.path.join(run_dir, "loss_analysis")
    os.makedirs(loss_dir, exist_ok=True)

    out_json = os.path.join(loss_dir, f"revalidate_{tag}_{stamp}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "ckpt_name": args.ckpt_name,
                "denoising_steps": int(cfg.validation.denoising_steps),
                "ensemble_size": int(cfg.validation.ensemble_size),
                "dtype": str(args.dtype),
                "processing_res": int(cfg.validation.processing_res),
                "match_input_res": bool(cfg.validation.match_input_res),
                "max_samples": int(args.max_samples),
                "save_vis": bool(args.save_vis),
                "vis_dir": vis_root,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    out_txt = os.path.join(loss_dir, f"revalidate_{tag}_{stamp}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2))
        f.write("\n")

    print("saved", out_json)
    print("saved", out_txt)


if __name__ == "__main__":
    main()
