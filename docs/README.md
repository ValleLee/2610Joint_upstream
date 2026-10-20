# JointTeacher 论文材料索引

本目录对应 **2026-10-02 16:46:57 启动、训练至 15000 iter** 的运行：
原始运行目录名为 `output/20261002_164657_train_rgbdepth_teacher_perimg`。
仓库中保留精选图、原始样例与评估证据；完整运行目录和模型权重在外部。

- [方法与实验设计](method.md)：任务、数据字段、扩散公式、网络结构、损失、训练与推理流程。
- [结果与图像](results.md)：完整验证指标、收敛曲线、每场景三帧的 RGB/深度对照、追加的 DA(rendered RGB) 四列拼图及配对误差。
- [指标原始文本](evidence/evaluation/) 与 [整理后的 CSV](evidence/validation_metrics.csv)。
- [本次运行配置快照](evidence/run_config.yaml)、[14000 步 scheduler 配置](evidence/scheduler_config_iter014000.json)、[运行日志](evidence/run_logging.log)与[TensorBoard 记录](evidence/tensorboard/)。
- [图像原件](figures/source/)：所选 6 帧的 GT RGB、原始渲染 RGB、处理后 RGB，DA(GT RGB)、DA(rendered RGB)、原始渲染深度和处理后深度的 NPY；还保留 DA 的原始可视化 PNG 与训练输出的 16-bit 深度 PNG。
- [独立彩色深度图](figures/panels/)：四列拼图里的每一格都可单独查看；拼图为 [花园](figures/garden_depth_vs_da.png) 和 [雕塑](figures/sculpture_depth_vs_da.png)。
- [DA 与 JointTeacher 配对比较](evidence/da_comparison_100.csv)：14000 步保存的 100 帧，含 [JSON 汇总](evidence/da_comparison_100_summary.json) 与 [可复算脚本](evaluate_da_comparison.py)。
- [来源与 SHA256](evidence/provenance.json)：每个原件的原路径、验证清单行号和哈希。
- [绘图脚本](build_figures.py)：从已复制的原件重建所有图。

重建图表：

```bash
python3 docs/build_figures.py
```

绘图脚本默认使用 `docs/figures/source` 中的本地原件，**无须模型权重或外部数据即可重建七张汇总图**。传入 `--refresh-sources` 才重新复制原始数据和运行输出；在另一台机器上可先设置 `JOINT_RUN_DIR` 与 `JOINT_PAIR_FILE` 指定外部路径。

重新计算 [100 帧 DA 对比](evidence/da_comparison_100.csv) 需要原始验证数据及 14000 步保存的 100 张预测 NPY：

```bash
python3 docs/evaluate_da_comparison.py \
  --run-dir /path/to/20261002_164657_train_rgbdepth_teacher_perimg \
  --pair-file /path/to/val_pairs.txt
```

已保存的 CSV、JSON、六帧原始样例及图像不依赖这些外部文件即可查阅。DA 生成脚本副本位于 [script/data/generate_da2_inverse_depth.py](../script/data/generate_da2_inverse_depth.py)；其运行仍需外部 DA-V2 代码和权重。

**论文表述注意**：训练主指标是固定扩散时间步上的 latent MSE，不是最终 RGB 的 PSNR/SSIM 或真实物理深度误差。渲染深度仅为展示取倒数并独立分位数归一化。新增四列图的第 1、3、4 列按每帧 DA(GT RGB) 共用色阶；DA(rendered RGB) 和最终模型输出深度都已对齐到该 teacher。100 帧比较显示 DA(rendered RGB) 在 78 帧深度伪标签误差更低；**这不能推断它在下游应用中也优于联合预测器**。详见 [比较口径与边界](results.md#与-darendered-rgb-的直接对照)。
