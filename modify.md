# 2026-10-02：参照另一台服务器 `0930jointR` 对齐联合训练逻辑

## 修改目的

当前服务器上的工程继续使用原有的大规模 K24 数据和训练配置，只将联合
RGB-depth 模型的初始化、训练损失和验证选模逻辑对齐到参考服务器上的
`0930jointR`（参考源码路径：`/data/lilinwei/3D/0930jointR`）。该参考路径
不属于当前服务器；当前服务器的数据、权重和工程路径见文末“当前服务器路径核对”。

联合任务保持为：

```text
condition = [latent(rendered RGB), latent(rendered depth)]
target    = [latent(GT RGB), latent(teacher depth)]
UNet in   = [condition, noisy target]  # 16 channels
UNet out  = predicted joint target     # 8 channels
```

## 修改文件

1. `src/trainer/marigold_depth_trainer.py`
2. `src/trainer/marigold_depth_trainer_earlystop.py`
3. `config/train_rgbdepth_teacher_perimg.yaml`
4. `script/train_rgbdepth_k24_r100_perimg.sh`
5. `tests/test_joint_teacher_local_contract.py`（新增回归测试）

原文件备份位于：

```text
.backup/20261002_022642/
```

## 具体修改

### 1. UNet 16 通道输入初始化

预训练 Marigold 的 8 通道输入权重按以下方式扩展：

```text
0:4    rendered RGB       <- pretrained RGB-condition weights
4:8    rendered depth     <- pretrained RGB-condition weights
8:12   noisy GT RGB       <- pretrained noisy-target weights
12:16  noisy teacher depth <- pretrained noisy-target weights
```

修改前，`noisy GT RGB` 的 8:12 通道为零初始化。修改后，两个 noisy target
分支都继承预训练 noisy-target 权重。

YAML 中的 `trainer.unet_in_channels: 12` 保持不变。远端 trainer 原有逻辑会
为新增的 RGB target 再加 4 通道，因此实际 UNet 输入仍为 16 通道。

### 2. UNet 8 通道输出初始化

预训练 4 通道输出头同时复制到两个联合输出分支：

```text
0:4  predicted GT RGB latent      <- pretrained output head
4:8  predicted teacher-depth latent <- pretrained output head
```

修改前，RGB 输出分支为零初始化，只有 depth 输出分支继承预训练权重。

### 3. 完整联合 latent diffusion loss

训练 loss 改为对完整 8 通道预测与目标直接计算 MSE：

```text
loss = MSE(model_pred[:, 0:8], diffusion_target[:, 0:8])
```

不再用 teacher-depth 有效区域 mask 遮挡联合 latent。RGB 和 depth 各四个
通道在完整 latent 网格上以相同权重参与 loss。

### 4. 联合 latent 验证与选模

验证使用固定随机种子和扩散中点 timestep，报告：

- `rgb_latent_mse`
- `depth_latent_mse`
- `joint_latent_mse = (rgb_latent_mse + depth_latent_mse) / 2`

主验证指标、最佳 checkpoint 和 early stopping 均改用
`joint_latent_mse`，目标为 minimize。启动脚本的 `MAIN_VAL_METRIC` 默认值也
同步改为 `joint_latent_mse`。

## 保留的远端配置

以下设置没有修改：

- 36,430 个训练样本和 715 个验证样本；
- 五列 manifest 格式以及读取第 1、2、3、5 列的逻辑；
- 15,000 个训练 iteration；
- 正式启动脚本的 effective batch 32、单卡 batch 4；
- YAML 原始 effective batch 64、单卡 batch 16 默认值；
- learning rate 3e-5；
- LR scheduler 总长 18,000、warmup 200、final ratio 0.01；
- 432x768 输入尺寸；
- 保存、验证、可视化、backup 和 early stopping 周期；
- 四卡 DDP、xFormers、数据预处理和 depth normalization；
- 已有训练输出和 checkpoint。

已有 15,000-iteration checkpoint 不会被此修改重写。新的初始化逻辑只在
从基础 Marigold checkpoint 启动新训练时生效；从旧联合 checkpoint resume
会继续使用 checkpoint 中已经训练好的权重。

## 验证

在远端正式 `do3` 环境中运行联合逻辑回归测试：

```text
test_conv_in_gives_both_noisy_targets_the_pretrained_noisy_weights ... ok
test_conv_out_gives_both_joint_targets_the_pretrained_head ... ok
test_joint_training_loss_uses_every_rgb_and_depth_latent_element ... ok
test_joint_validation_reports_rgb_depth_and_equal_channel_joint_mse ... ok
test_remote_scale_and_schedule_defaults_are_preserved ... ok
```

补丁同步到正式目录后已完成以下验证：

- 远端 `do3` 环境回归测试：5/5 通过；
- 两个 trainer 文件与测试文件的 `py_compile`：通过；
- 正式启动脚本 `bash -n`：通过；
- 配置审计：15,000 iter、batch、LR schedule、分辨率和训练/验证样本数均保持原值；
- YAML 与启动脚本相对备份均只有 `MAIN_VAL_METRIC` 默认值发生变化。

本次没有启动新训练。

## 后续修复：恢复定期可视化文件写入

联合 latent 验证覆盖了原有的 `validate_single_dataset`，但最初没有处理
`save_to_dir`，导致定期可视化只创建目录、不保存预测图。现在有保存目录时
调用原有的图像推理与保存路径，写入预测 RGB PNG、深度 PNG 和深度 NPY；
有保存目录时仍返回 `rgb_latent_mse`、`depth_latent_mse` 和
`joint_latent_mse`，常规验证继续用 `joint_latent_mse` 选模。新增一项
回归测试检查三类文件实际生成，且保存操作不改变返回的验证指标。

## 当前服务器路径核对

`/data/lilinwei/3D/0930jointR` 仅表示参考服务器上的源码位置；当前工程
使用本服务器的 `/data1/lilinwei` 和 `/data2/caoxiaodata/data`。已逐项核对
本工程 ratio100 的训练 36,430 行、验证 715 行：五列文件均存在，前四列
与本服务器的 Plan-D 源列表一致，第五列均指向同场景同帧的 `images_4`。
本服务器的 Marigold 基础权重位于
`/data1/lilinwei/checkpoints/marigold-depth-v1-0`。

恢复脚本原有的 `REPLACE_ME` 默认 checkpoint 路径已移除；恢复时必须显式
提供当前服务器上实际存在的 `RESUME_RUN` 路径。
