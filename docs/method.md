# JointTeacher：目标、方法与实验设计

## 1. 研究目标

本实验研究一个**条件联合生成**任务：给定低质量的渲染 RGB 图和与它对应的渲染深度图，同时预测接近 GT RGB 的图像，以及接近从 GT RGB 计算出的 DA 深度 teacher 的深度图。RGB 和深度共享一次扩散去噪过程，因此两个输出可以利用同一场景的外观与几何线索。这里的“深度 GT”是 **DA 伪标签**，并非传感器真值或可直接视为米制的绝对深度。

输入与输出的来源及代码证据：

| 清单列 | 内容 | 在当前模型中的作用 |
|---|---|---|
| 1 | 渲染 RGB，`lr/24/train_24/*.png` | RGB 条件 |
| 2 | 渲染深度，`rendered_depth/*.npy` | 深度条件 |
| 3 | 从 GT RGB 得到的 DA 数组，`DAV2_depth_inverse/.../rgb_gt/depth_npy/*.npy` | 深度 teacher 目标 |
| 4 | 从渲染 RGB 得到的 DA 数组，`rgb_rendered/depth_npy/*.npy` | **当前 Dataset 不读取**；用于事后基线比较 |
| 5 | `gaussian_splat/images_4/*.png` | GT RGB 目标 |

五列含义由 [数据读取代码](../src/dataset/rgb_invdepth_teacher_dataset.py) 中 `_get_data_item` 的索引直接确定。训练清单有 **36430** 行、**113** 个场景，其中 `*_ref` **2712** 行；验证清单有 **715** 行、**2** 个场景，其中 `*_ref` **48** 行。训练与验证所用清单分别是 [train_pairs.txt](../data/pairs/k24_ratio100_val2_ignore/train_pairs.txt) 和 [val_pairs.txt](../data/pairs/k24_ratio100_val2_ignore/val_pairs.txt)。因此，这次实验**包含 ref 帧**。

### 深度表示的关键细节

运行时 [配置快照](evidence/run_config.yaml) 对条件与 teacher 都设置了 `*_from_depth: true`、`*_representation: depth`。数据集的 `_should_invert(true, "depth")` 返回 `false`，所以第二、第三列 NPY **读取时均未取倒数**，之后各自逐图归一化。[DA 原始生成脚本](../script/data/generate_da2_inverse_depth.py) 明确指出第三、四列是 DA-V2 原始输出，**近处数值大**，生成时未取倒数。第二列渲染深度在抽查帧上与 DA 呈相反的数值排序；这影响直接展示时的颜色方向，但本次训练确实按配置读入第二列原值。结果图为便于比较，对第二列**仅在绘图时**取倒数，并在图注中标出。DA 是相对深度伪标签，不是米制真值。

## 2. 数据预处理与条件编码

两路 RGB 按数据集流程缩放到 `432×768`，数值映射至 `[-1,1]`。渲染深度与 DA teacher 数组分别使用有效值 `finite & >0` 计算每图的 1% 与 99% 分位数，线性映射并裁剪到 `[-1,1]`；第 2、3 列采用相同**规则**，但每张数组独立计算分位数。代码见 [dataset](../src/dataset/rgb_invdepth_teacher_dataset.py) 与 [depth_transform](../src/util/depth_transform.py)。

记冻结 VAE 的确定性编码为 \(E(x)=s\,\mu_{\mathrm{VAE}}(x)\)，其中运行时 latent scale \(s=0.18215\)。单通道深度先复制为 3 通道，再送入同一 VAE，见 [pipeline](../marigold/marigold_depth_pipeline.py) 的 `encode_rgb` 和 `encode_depth`。四个 4 通道 latent 为

\[
c_r=E(I_{\rm render}),\qquad c_d=E_\mathrm{depth}(N(D_{\rm render})),\qquad
y_r=E(I_{\rm GT}),\qquad y_d=E_\mathrm{depth}(N(D_{\rm DA})).
\]

其中 \(N\) 是前述逐图归一化。静态条件为 \(c=[c_r,c_d]\in\mathbb R^{8\times h\times w}\)，联合目标为 \(y_0=[y_r,y_d]\in\mathbb R^{8\times h\times w}\)。

## 3. 联合 latent diffusion

基础模型是预训练 Marigold。训练仅更新 UNet；VAE 和文本编码器冻结，文本条件固定为空 prompt 的 embedding。每个样本独立从 1000 个 DDPM 时间步中均匀抽取 \(t\)，向联合目标加噪：

\[
y_t=\sqrt{\bar\alpha_t}\,y_0+\sqrt{1-\bar\alpha_t}\,\epsilon.
\]

训练的 \(\epsilon\) 是多尺度噪声，强度基准为 0.8，并按 \(t/1000\) 退火；具体构造在 [multi_res_noise.py](../src/util/multi_res_noise.py)。基础 scheduler 在本次训练中的 `prediction_type` 为 **`v_prediction`**，故实际监督目标是

\[
v_t=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,y_0,
\qquad
\hat v=\epsilon_\theta([c_r,c_d,y_t],t,e_{\varnothing}).
\]

此处 \(\epsilon_\theta\) 记号沿用扩散模型惯例，但本次输出是 velocity \(\hat v\)，并非直接的 RGB、深度或纯噪声。UNet 输入为 **16 通道**：4 通道渲染 RGB latent、4 通道渲染深度 latent、8 通道带噪的联合目标 latent；输出为 **8 通道**：前 4 通道对应 RGB，后 4 通道对应深度。代码见 [trainer](../src/trainer/marigold_depth_trainer.py) 的初始化和训练循环。

预训练 UNet 原有 `conv_in` 为 8 通道、`conv_out` 为 4 通道。本实验的初始化把原 RGB 条件权重复制到两个条件分支，把原 noisy target 权重复制到两个带噪目标分支，并把原输出头复制到 RGB 与深度两个输出分支；之后全部由联合训练更新。具体权重切片见 `MarigoldDepthTrainer._replace_unet_conv_in/_replace_unet_conv_out`。YAML 的 `trainer.unet_in_channels: 12` 会在 trainer 内再加 4，运行时实际值为 16。

## 4. 损失与优化

对完整 8 通道 latent 做等权 MSE：

\[
\mathcal L_{\rm joint}
=\frac{1}{8hw}\sum_{k=1}^{8}\sum_{u,v}
\left(\hat v_{k,u,v}-v_{t,k,u,v}\right)^2
=\frac{\mathcal L_{\rm RGB}+\mathcal L_{\rm depth}}{2}.
\]

`reduction: mean` 对 batch、通道和空间位置求平均。代码虽构造了深度有效区域 mask，但本次 `_compute_joint_latent_loss` **不使用 mask**；RGB 与深度各 4 通道在全 latent 网格参与损失。mask 仍用于原始深度的归一化和可视化评估。训练代码见 [marigold_depth_trainer.py](../src/trainer/marigold_depth_trainer.py)，本次对齐说明见 [modify.md](../modify.md)。

| 参数 | 本次运行值 |
|---|---:|
| 最大 optimizer iteration | 15000 |
| 学习率 / 优化器 | `3e-5` / Adam |
| LR 调度 | IterExponential，总长 18000，warmup 200，末端比例 0.01 |
| 全局有效 batch | 32（4 GPU × 每卡 4 × 累积 2） |
| 输入尺寸 | 432×768 |
| AMP / activation checkpointing | fp16 / 开启 |
| 训练噪声调度 | DDPM，1000 步，`rescale_betas_zero_snr=true`、`timestep_spacing=trailing` |
| 数据增广 | 水平翻转概率 0.5 |
| 验证 / 保存 | 前期每 500 步、6000 步后每 1000 步验证；checkpoint 周期 1000 步 |
| early stopping | 10000 步后，patience 5 次、`min_delta=5e-5` |

参数以本次 [运行配置快照](evidence/run_config.yaml)、[运行日志](evidence/run_logging.log) 和 [启动脚本](../script/train_rgbdepth_k24_r100_perimg.sh) 为准，不应把 YAML 模板的默认 batch 64 误写成本次实际 batch。

## 5. 验证与图像生成

主验证并不完整执行 4 步反向扩散。它对每张验证图固定 \(t=500\)，使用由 `2024 + batch_index` 决定的 Gaussian 噪声构造 \(y_t\)，然后计算单步预测的 `rgb_latent_mse` 与 `depth_latent_mse`；`joint_latent_mse` 是二者算术平均，对 715 张样本求均值。指标越低越好，用于训练中的模型选择与 early stopping。实现见 [EarlyStop trainer](../src/trainer/marigold_depth_trainer_earlystop.py)。

可视化另走 [MarigoldDepthPipeline](../marigold/marigold_depth_pipeline.py)：从 8 通道 Gaussian latent 开始，利用渲染 RGB/深度条件做 **4 步 DDIM**，前 4 通道 VAE 解码为 RGB；后 4 通道 VAE 解码为 3 通道后取均值，得到单通道深度并映射到 `[0,1]`。可视化保存路径又将该预测深度逐图做最小二乘 scale/shift 对齐到 DA teacher，之后裁剪为正数并保存 NPY。因此结果图中的预测深度是**对齐后的可视化输出**，并不证明模型直接输出绝对尺度深度。保存逻辑见 [trainer](../src/trainer/marigold_depth_trainer.py) 的 `validate_single_dataset`。

## 6. 论文中可直接使用的边界陈述

本次结果支持“联合 RGB–DA 深度条件扩散训练显著降低固定时间步的 joint latent 去噪误差，并在多帧例子中从早期噪声收敛到具有场景结构的输出”。本次记录**尚不能**支持“比原始渲染 RGB 在标准图像质量指标上更优”或“恢复真实米制深度”的结论：没有同集基线的 PSNR/SSIM 或物理深度真值，且定性结果仍存在细节变软与局部深度偏差。完整数据和图见 [results.md](results.md)。
