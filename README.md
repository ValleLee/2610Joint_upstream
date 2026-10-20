# JointTeacher：RGB 与 DA 深度联合生成实验

本目录整理了 `DepthDenoiserJointTeacher` 在 **2026-10-02 至 2026-10-03** 完成的
一次训练。模型以渲染 RGB 和渲染深度为条件，联合预测 GT RGB 与由 GT RGB
生成的 DA-V2 相对深度伪标签。训练完成 **15,000 iter**；本次记录中
`joint_latent_mse` 最低的验证点是 **14,000 iter**。训练和验证清单分别为
36,430 与 715 对，均保留了 `_ref` 行。

本目录供复现实验方法、核对配置和阅读结果使用。模型权重、基础 Marigold
checkpoint 和完整训练输出由使用者在本地提供；精选图像、指标、原始可视化
样例和日志证据已随文档保留。

## 目录

| 路径 | 内容 |
|---|---|
| `src/`, `marigold/` | 联合训练 dataset、trainer、pipeline 与工具代码 |
| `config/` | 训练 YAML 及引用的基础 YAML |
| `script/` | 训练、恢复、数据清单生成和路径迁移脚本 |
| `data/pairs/k24_ratio100_val2_ignore/` | 本次运行使用的原始五列 train/val 清单 |
| `docs/method.md` | 方法、配置与训练流程 |
| `docs/results.md` | 验证曲线、RGB/深度拼图、DA 直接估计对比及局限 |
| `docs/figures/` | 七张汇总图、单张彩色深度图和所选六帧的原始文件 |
| `docs/evidence/` | 运行配置快照、验证原始文本、逐帧比较、运行日志和 TensorBoard 记录 |
| `environment/` | 已验证环境的主要 Python 依赖版本 |
| `tests/` | 联合网络通道、损失及可视化保存行为检查 |

本次实际运行参数以 [配置快照](docs/evidence/run_config.yaml) 为准；可修改的
训练模板是 [train_rgbdepth_teacher_perimg.yaml](config/train_rgbdepth_teacher_perimg.yaml)，
实际启动环境变量见 [训练脚本](script/train_rgbdepth_k24_r100_perimg.sh)。例如，
本次全局 batch 是 **32**（4 GPU × 每卡 4 × 梯度累积 2），而模板在没有
启动脚本覆盖时的默认值不同。

## 环境与数据

已验证环境为 Python 3.10.19、PyTorch 2.4.1+cu118。先按目标机器 CUDA
环境安装相应 PyTorch 与 torchvision，再安装
[`environment/requirements.txt`](environment/requirements.txt) 中的依赖。
基础权重需放在 `checkpoints/marigold-depth-v1-0/`，或设置 `BASE_CKPT_DIR`
指向包含 `checkpoints/marigold-depth-v1-0/` 的目录。权重不在本目录内。

五列数据清单保留了原机器的绝对路径。迁移机器时应按
[数据说明](data/README.md) 重写路径并检查文件存在性，然后设置 `PAIRS_DIR`。
第 4 列 DA(rendered RGB) 用于事后对比，当前训练 dataset 使用第 1、2、3、5 列。

## 训练

激活已安装依赖的 Python 环境后，在项目根目录运行。例如重现四卡配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
EFFECTIVE_BATCH_SIZE=32 \
MAX_TRAIN_BATCH_SIZE=4 \
MAX_ITER=15000 \
PAIRS_DIR="$PWD/data/pairs/local" \
BASE_CKPT_DIR="$PWD" \
PYTHON_BIN=python3 \
bash script/train_rgbdepth_k24_r100_perimg.sh
```

`output/` 和 `logs/` 会在本地生成且已由 `.gitignore` 排除。恢复训练的入口是
[`script/resume_rgbdepth_k24_r100.sh`](script/resume_rgbdepth_k24_r100.sh)，需设置
`RESUME_RUN` 为本地完整 checkpoint 目录。启动脚本在整理副本中只改动了
Python、GPU 和权重根路径的**默认选择方式**，训练 Python 代码与 YAML
超参数保留自原实验。

## 结果与复核

- [实验方法](docs/method.md)
- [结果分析与图像](docs/results.md)
- [来源及复制范围](docs/SOURCE_PROVENANCE.md)
- [验证指标 CSV](docs/evidence/validation_metrics.csv) 与 [原始记录](docs/evidence/evaluation/)
- [100 张 DA 对比逐帧 CSV](docs/evidence/da_comparison_100.csv)

在保存的 100 张验证帧上，以 DA(GT RGB) 为伪标签并对齐尺度后，
DA(rendered RGB) 在 **78/100** 帧上的深度误差更低。这个比较不评价真实
物理深度，也不说明 DA 在下游应用中一定优于联合预测器；下游应用仍需按
相同任务流程和指标独立评估。

已复制的图像原件足以离线重建汇总图：

```bash
python3 docs/build_figures.py
python3 -m unittest discover -s tests -v
```

重新计算 100 张逐帧 DA 对比需要原始数据文件及 14,000 iter 保存的 100 张
预测 NPY；请参阅 [`docs/README.md`](docs/README.md) 的说明。后续上传前可运行
`python3 script/verify_package.py` 检查文件完整性和权重排除规则。
