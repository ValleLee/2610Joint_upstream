# 来源与整理范围

## 原始实验

- 训练工程：`/data1/lilinwei/code/Ad23e/npydepthdenoiser/DepthDenoiserJointTeacher`
- 本次运行：`output/20261002_164657_train_rgbdepth_teacher_perimg`
- 训练结束：2026-10-03 09:12:22；运行到 15,000 iter
- DA-V2 数据生成脚本来自
  `/data1/lilinwei/Depth-Prior-Optimization-for-SVGS/code/data_maker/generate_da2_inverse_depth.py`

工程本身是从此前的 DepthDenoiser 代码派生的联合 RGB/深度训练版本。
本目录只整理这次 JointTeacher 运行及其结果分析；原工程仍保存在上面的路径。

## 保留内容

原工程的 `src/`、`marigold/`、`config/`、`script/`、`tests/`、
`data/pairs/k24_ratio100_val2_ignore/` 和 `docs/` 已复制到本目录。
六个图示验证帧的 60 份 RGB、DA 和模型输出原件，以及由它们生成的汇总图、
独立彩色深度图、逐次验证文本与 100 帧 DA 对比 CSV 均在 `docs/` 内。
各原件的原路径和 SHA256 位于
[`evidence/provenance.json`](evidence/provenance.json)。
逐文件的原工程哈希、副本哈希及是否逐字节相同记录于
[`evidence/source_copy_manifest.json`](evidence/source_copy_manifest.json)。

另从本次运行目录复制了 `logging.log` 为
[`evidence/run_logging.log`](evidence/run_logging.log)，并保留原始
[TensorBoard event](evidence/tensorboard/)。本次实际解析后的配置快照为
[`evidence/run_config.yaml`](evidence/run_config.yaml)。

## 整理副本中的改动

- `script/train_rgbdepth_k24_r100_perimg.sh` 和 `script/resume_rgbdepth_k24_r100.sh`
  将原服务器固定的 Conda/Python、GPU 和 checkpoint 根路径改为环境变量默认值；
  核心训练 Python 代码和 YAML 超参数没有改动。
- `script/build_joint_pairs.py` 加入外部 source/GT 根路径参数；新增
  `script/remap_pair_paths.py`，便于迁移五列清单中的历史绝对路径。
- `docs/build_figures.py` 可通过 `JOINT_RUN_DIR`、`JOINT_PAIR_FILE` 指定外部
  原始数据；本地源样例足以直接重建汇总图。`docs/evaluate_da_comparison.py`
  加入外部运行目录和验证清单参数。
- 新增环境依赖表、包完整性检查脚本，以及 Git 忽略规则。
- DA-V2 原始生成脚本被单独复制到 `script/data/`，用于解释第 3、4 列
  NPY 的方向约定；运行该脚本仍需要外部 DA-V2 仓库和 checkpoint。

完整训练 `output/`、全部迭代的 4.7 GB 可视化流、约 46 GB checkpoint、
基础 Marigold 权重及外部训练/验证数据没有打包。已保留论文分析直接引用的
图、原始样例、指标、日志和配置。运行目录中未复制的文件在原始工程中仍可
查阅；这份整理副本没有修改或删除原始工程。
