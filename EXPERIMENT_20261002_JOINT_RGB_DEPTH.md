# 2026-10-02 联合 RGB–depth 训练记录

本文记录这次**已经完成**的训练，而不是启动脚本在没有环境变量覆盖时的默认值。原始运行目录名为 `output/20261002_164657_train_rgbdepth_teacher_perimg/`；整理副本保留了[运行配置快照](docs/evidence/run_config.yaml)、[日志](docs/evidence/run_logging.log)及[验证原始文本](docs/evidence/evaluation/)，完整输出目录及权重由使用者在外部保存。此前的代码修改过程见 [modify.md](modify.md)。

## 任务与数据

任务是在同一扩散模型中联合恢复同帧 RGB 和 depth：

```text
条件：渲染 RGB + 渲染 depth
目标：images_4 中的 GT RGB + 在 GT RGB 上预先生成的 DA teacher depth
推理输出：预测 RGB 图 + 预测 depth 图
```

使用 K=24、ratio100、val2、ignore 划分。训练列表 `data/pairs/k24_ratio100_val2_ignore/train_pairs.txt` 有 **36,430** 行，验证及可视化列表 `val_pairs.txt` 有 **715** 行。每行五列，顺序如下：

| 列 | 内容 | 当前训练用途 |
| --- | --- | --- |
| 1 | 渲染 RGB | RGB 条件 |
| 2 | 渲染 depth | depth 条件 |
| 3 | 基于 GT RGB 生成的 DA teacher depth | depth 目标 |
| 4 | 基于渲染 RGB 生成的 DA depth | 当前数据集不读取 |
| 5 | 同场景同帧的 `gaussian_splat/images_4` GT RGB | RGB 目标 |

读取逻辑见 `src/dataset/rgb_invdepth_teacher_dataset.py::_get_data_item`。样本文件使用本机 `/data1/lilinwei` 和 `/data2/caoxiaodata/data` 路径。GT RGB 会缩放到渲染 RGB 的网格；本次训练、验证和可视化配置的尺寸都是 **432×768**。depth 条件和 teacher 目标按配置 `condition_from_depth=true`、`teacher_from_depth=true`、`condition_representation=depth`、`teacher_representation=depth` 读取，然后使用 `scale_shift_depth` 做每张图的归一化：范围 `[-1, 1]`、分位数 `0.01`、裁剪开启。这里描述的是代码实际执行的解释方式；目录名中的 `depth_inverse` 不会自动改变这些配置。

## 模型初始化

从本机 `/data1/lilinwei/checkpoints/marigold-depth-v1-0` 加载 Marigold。该次运行从基础权重开始，启动日志显示扩展输入、输出层后直接开始第 1 次迭代；没有从上次联合训练 checkpoint 接续。VAE、文本编码器冻结，整个 UNet 可训练；文本条件使用空提示词 embedding。

预训练 UNet 原本接收 `[RGB 条件(4), 带噪 depth(4)]` 共 8 通道。本次实际接收 16 通道，输入卷积权重这样复制：

| 新输入通道 | 内容 | 从 Marigold 输入卷积复制 |
| --- | --- | --- |
| 0:4 | 渲染 RGB latent | 原 RGB 条件 0:4 |
| 4:8 | 渲染 depth latent | 原 RGB 条件 0:4 |
| 8:12 | 带噪 GT RGB latent | 原带噪目标 4:8 |
| 12:16 | 带噪 teacher depth latent | 原带噪目标 4:8 |

原输出卷积的 4 通道权重及 bias 分别复制到新输出的 **0:4 RGB** 和 **4:8 depth** 两个分支。UNet 其余参数沿用 Marigold。配置中 `trainer.unet_in_channels: 12` 沿用旧值，trainer 会额外加 4，因此**实际是 16**；以运行时模型和代码为准。实现见 `src/trainer/marigold_depth_trainer.py::_replace_unet_conv_in/_replace_unet_conv_out`。

## 一次训练迭代

1. 冻结的 VAE 把渲染 RGB、渲染 depth、GT RGB 和 teacher depth 分别编码为 4 通道 latent。单通道 depth 先复制成 3 通道再经同一 VAE 编码器；编码使用 posterior mean，latent scale factor 为 `0.18215`。
2. 将目标拼成 `x₀ = [latent(GT RGB), latent(teacher depth)]`，共 8 通道。每张样本随机采样扩散时间步，对整个 `x₀` 加噪得到 `xₜ`。训练使用由 Marigold scheduler 配置构造的 DDPM noise scheduler，`rescale_betas_zero_snr=true`、`timestep_spacing=trailing`；多分辨率噪声强度 `0.8`，并随时间步退火。
3. UNet 输入为 `[latent(渲染 RGB), latent(渲染 depth), xₜ]`，共 16 通道，输出 8 通道。该权重的 `prediction_type` 为 **`v_prediction`**，因此模型学习预测扩散速度 `v`，而不是直接回归干净图像或像素深度。
4. 唯一训练损失是完整 8 通道上的 `MSE(预测 v, 目标 v)`，`reduction=mean`。RGB 与 depth 各占 4 通道，在同一 latent 网格上等权参与。配置保留 `gt_mask_type: valid_mask_raw`，但训练损失**没有按该 mask 筛选像素**；该 mask 在这段训练逻辑中用于异常诊断。没有额外的 RGB 像素损失或 depth RMSE 损失。

对应代码是 `src/trainer/marigold_depth_trainer.py::train`、`encode_rgb`、`encode_depth` 和 `_compute_joint_latent_loss`。

## 推理、验证与可视化

推理时只有渲染 RGB 和渲染 depth 两个条件；另外创建随机的 8 通道目标 latent，使用 **4 步 DDIM** 逐步去噪。最终前 4 通道经 VAE 解码为 RGB，后 4 通道经 VAE 解码并对三个输出通道取均值，得到单通道 depth。见 `marigold/marigold_depth_pipeline.py::single_infer`。

**选模验证与上述图像推理分开**：`MarigoldDepthTrainerEarlyStop.validate_single_dataset` 用固定种子 `2024 + batch_index` 和固定中点时间步，对验证目标加噪后只做一次 UNet 预测，分别报告 `rgb_latent_mse`、`depth_latent_mse`，并令 `joint_latent_mse = (rgb_latent_mse + depth_latent_mse) / 2`。主指标为后者，越低越好；保存最佳 checkpoint 使用严格变小的条件。这个值是 latent 扩散预测误差，不能直接理解为生成 RGB/深度图的 PSNR、SSIM 或 RMSE。

定期可视化则从验证列表选取前 50 条和后 50 条，走完整的解码推理，并在 `visualization/iter_*/RGB+DepthCond_PerImg_Vis/` 写入预测 RGB PNG（`*_rgb.png`）、预测 depth PNG（`*.png`）和浮点 depth NPY（`*.npy`）。`validation.denoising_steps: 4` 用在此图像推理路径，不用于上述单时间步的选模指标。图像评测配置仍列有 `abs_relative_difference`、`rmse_linear`，它们不是此次最佳权重的选择依据。

## 本次实际运行配置

| 项目 | 实际值 |
| --- | --- |
| GPU | `CUDA_VISIBLE_DEVICES=4,5,6,7`，4 个 DDP rank |
| 全局有效 batch | 32；每卡 micro-batch 上限 4；梯度累积 2 步 |
| 精度与内存 | `fp16` AMP、gradient checkpointing、xFormers |
| 优化器 | Adam，只优化 UNet；初始学习率 `3e-5` |
| 学习率调度 | `IterExponential`，配置总长 18,000 次、warmup 200 次、final ratio `0.01`；本次在 15,000 次结束 |
| 训练上限 | 15,000 次 optimizer update；随机种子 2024 |
| 验证与可视化 | 1–6,000 次每 500 次一次；之后每 1,000 次一次 |
| 存档 | `latest` 周期 1,000 次；backup 周期 2,000 次；验证改善时保存对应 `iter_*` 权重 |
| Early stopping | 从 10,000 次开始，patience 5 次验证、`min_delta=5e-5`；本次未提前停止 |

本次是 4 卡运行；`script/train_rgbdepth_k24_r100_perimg.sh` 未覆盖时的默认值是 2 卡，不能用该默认值代替此次实际配置。实际覆盖值保存在运行目录 `config.yaml` 的 `runtime.env`。

## 结果与文件位置

- 训练于 **15,000 次**正常结束。最佳选模指标在 **14,000 次**：`joint_latent_mse=0.2085847576`，其中 RGB `0.2776838548`、depth `0.1394856603`。最佳权重位于 `output/20261002_164657_train_rgbdepth_teacher_perimg/checkpoint/iter_014000/`。
- 15,000 次终点权重位于 `checkpoint/iter_015000/`，对应 `joint_latent_mse=0.2085886269`。它与最佳值非常接近，但不是此次选模最优。
- 逐次验证值见 `evaluation/eval-RGB+DepthCond_PerImg_Val-iter*.txt`；完整训练日志见 `logging.log`；可视化见 `visualization/`；`checkpoint/latest/` 用于恢复训练。

这些指标只适合在**相同数据、验证定义与时间步**下比较。与 `output/20261001_052455_train_rgbdepth_teacher_perimg/` 的旧验证数值采用不同指标时，不应直接用数值大小判断优劣；比较最终图像质量应在统一图像评测流程中进行。
