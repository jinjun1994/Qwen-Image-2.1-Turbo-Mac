# 性能优化与实验记录

[返回中文 README](../README.md) · [English README](../README.en.md)

以下完整保留原 README 从“常驻引擎优化（2026-10-26）”起的内容。文中 `reports/`、`outputs/` 为开发时的本地报告和产物，未随源码发布；部分实验未启用，历史测量不代表当前版本。当前桌面版交流电实测摘要见 [desktop-ac.json](benchmarks/desktop-ac.json)。

## 常驻引擎优化（2026-10-26）

服务启动时一次性加载模型；页面右上角显示“模型预加载中 / 模型常驻 · 已就绪”。加载完成后，即使没有生成任务，模型也继续占用内存，这是节省后续加载过程的预期行为。更换模型或 tokenizer 后需重启服务。

原先的每张图启动新进程已替换为单推理线程；仍只允许一个生成任务。提示词特征最多缓存 4 条，DiT 前缀缓存由 MFLUX 限制为最近 2 个嵌入，避免无限保留历史提示词。改变分辨率时按形状重新建立相应缓存。

初次验证（M3 Pro 36GB，512×512、8 步、同一茶壶提示词及 seed 42）：旧版 39.7 秒；常驻首次请求约 33.9 秒，缓存命中后约 29.8 秒。两个常驻结果与旧版基准逐像素一致。模型加载约 9–10 秒，在服务启动时单独支付，**不计入后续任务的 `total_seconds`**；初次打开服务至首图的等待时间应将启动加载算入。运行 JSON 中 `engine_load_seconds` 记录这笔启动成本，`prompt_cache_hit` 记录缓存是否命中。`mlx_peak_memory_bytes` 在常驻模式下是引擎进程生命周期内的分配器峰值，不能当成每张图独立峰值。

代表性旋转微基准显示：Python 蝶形组合比现有稠密旋转更慢；自定义 Metal 蝶形在该尺寸收益接近测量波动，故未替换默认运算。合并 QKV/FFN 的独立微基准有约 5% 收益，但完整 512 生图约 30.2 秒，未优于未融合的约 29.8 秒，且数值舍入改变了像素，因此未合入。标准模式保持原有推理数学和 8 步日程。

最终同机验证：

| 分辨率 | 原版每图重载 | 常驻缓存命中 | 延迟下降 | 图像比较 |
|---|---:|---:|---:|---|
| 512×512 | 39.7s | 29.814s | 约 25% | 逐像素一致 |
| 1024×1024 | 137.333s | 127.664s | 约 7% | 逐像素一致 |

常驻权重准备完成时 MLX 活跃分配约 15.06GiB；512 连续运行时进程峰值约 20.12GiB，切换到 1024 后进程峰值约 29.46GiB。1024 在本机 36GB 内存上已验证，内存余量较小，不能把该配置推广为小内存设备默认值。取消真实任务后再次生成及改变分辨率均已验证，模型没有重新加载。测量为单机单次/少量重复结果，受其他应用和热状态影响。

可以在已启动的 WebUI 上重复测量：

```sh
.venv/bin/python scripts/benchmark_resident.py --width 512 --height 512 --runs 2
```

报告写入 `reports/resident-benchmark.json`。首次缓存是否命中取决于服务之前的使用情况，报告会逐次记录，不将已有缓存伪装成冷启动。停止服务并重启可重新测量模型启动成本。

## 页面控制推理服务与快速预览

页首新增 **启动推理服务 / 关闭推理服务** 按钮。这里控制的是模型引擎；网页服务保留运行，才能从页面重新启动。关闭时如有任务，会先在 GPU 运算检查点取消任务，再清除编译函数引用、前缀缓存、提示词缓存和模型权重。实际测试关闭后 MLX 活跃分配显示 0.000GiB（不等于 Python 进程总 RSS 为零），重新启动后恢复约 15.06GiB 常驻权重。页面显示加载、就绪、释放、已关闭和失败状态。关闭后不能提交生成；启动完成后恢复。退出整个网页服务仍使用终端 Ctrl+C。

“采样步数”提供三档，**默认仍为标准 8 步**：

| 模式 | 日程 | 512×512 本机实测 | 用途 |
|---|---|---:|---|
| 标准 8 步 | 原始 9 个 sigma 节点 | 缓存命中约 29.8s | 最终出图 |
| 快速 6 步（实验） | 对原始 sigma 曲线等距插值得到 7 节点 | 22.3s | 较快试图，画质可能变化 |
| 极速预览 4 步（实验） | 对原始 sigma 曲线等距采样得到 5 节点 | 15.7s | 快速看构图，不能替代 8 步质量 |

1024×1024 的 4 步预览单次实测为 69.219s，其中文本编码 1.404s、去噪 59.155s、VAE 解码 8.648s；此前同机常驻 8 步约 127.664s。该次提示词缓存未命中，未计入服务启动加载时间。

快速模式通过减少模型调用次数提速，**不是同等画质下的算子加速**；相同 seed 的构图、纹理和文字可能改变。已目视检查 512 的 6/4 步及 1024 的 4 步茶壶示例；这不是广泛画质验证。标准 8 步与旧版基准仍逐像素一致。

CLI 也支持 `mqwen generate --steps 4 ...`；复测示例：

```sh
.venv/bin/python scripts/benchmark_resident.py --steps 4 --runs 2 --report reports/preview-benchmark.json
```

报告与 PNG 元数据均记录实际步数及完整 sigma 日程。减少步骤不会改变文本编码器、VAE 或权重精度。本次还测试了 FP16/量化投影合并，但未观察到足够的完整生成收益，相关实验没有作为正式推理路径。

## 继续优化：物理驻留与跨步特征缓存

引擎默认设置 MLX 的 Metal 物理驻留额度为 `min(系统建议额度, 物理内存的 60%)`，M3 Pro 36GiB 上为 21.60GiB。它是驻留上限，不是额外分配，也不保证整个系统绝不换页；关闭引擎时恢复原额度。旧系统不支持时记录日志并继续。需要对比或给其他应用腾出更多可分页内存时，可用 `mqwen web --wired-memory off ...`。运行元数据记录 `wired_limit_bytes`。驻留额度不改变推理数学，测试图与原基准逐像素一致。

WebUI 新增 **跨步特征缓存（实验，8 步）**，默认关闭。CLI 使用 `mqwen generate --steps 8 --step-cache ...`。它保留原始 8 步 sigma 日程，以第一层输出的相对 L1 变化判断是否复用后续 31 层的残差（阈值 0.08），输出调制始终重新计算。前两步、最后两步强制完整计算，禁止连续复用；每张图重新建立缓存，取消或失败后恢复原执行函数。这是近似计算，可能改变细节、文字及构图，不能视为无损优化，也不承诺每个提示词都能命中缓存。不能与 4/6 步同时使用。

运行 JSON 的 `step_cache` 记录阈值、变化指标、实际复用步号和完整执行次数。未开启时 `enabled=false`，走原有标准执行路径。复测示例：

```sh
.venv/bin/python scripts/benchmark_resident.py --steps 8 --step-cache --runs 2 --report reports/step-cache-benchmark.json
```

本轮还比较了 VAE 全图编译、普通分配器缓存从 512MiB 提高到 2GiB、量化分组大小、FP16 和展开权重 GEMM。单独 VAE 编译曾有收益，但在完整常驻模型中未显示稳定优势；2GiB 分配器缓存反而变慢。这些实验均未替换默认路径。单机有内存压力和运行波动，需区分热运行、首次编译与提示词缓存命中情况。

本轮实际 WebUI 请求对比（M3 Pro 36GiB，512×512，8 步；每项单次测量）：

| 提示词 | 标准路径 | 开启跨步缓存 | 延迟下降 | 实际复用步骤 |
|---|---:|---:|---:|---|
| 窗边红色茶壶，seed 42 | 28.942s | 23.865s | 约 18% | 3、5 |
| 日落水彩灯塔，seed 7 | 27.884s | 20.759s | 约 26% | 3、5 |

茶壶缓存请求包含该路径的首次编译，标准茶壶为热运行；灯塔两次文本提示词缓存命中情况不同。完整元数据在 `reports/optimization-verification.json` 中记录，不能将这两组结果当作严格统计基准。已目视检查两组缓存图；茶壶对标准图的像素 MAE 约 10.9/255，说明输出确有变化，该数值不代表感知质量评分。实际取消缓存任务后再次标准生成，与旧版基准逐像素一致。

1024×1024 茶壶的 8 步缓存模式已实测生图：103.356s（去噪 96.654s、VAE 6.682s），仅第 3 步命中复用；此前常驻标准模式为 127.664s。两次不是同轮交替对照，只能作为参考，不能把约 19% 差值全归因于缓存。此轮引擎进程 MLX 分配峰值约 28.77GiB。缓存请求后关闭引擎，实际日志显示活跃分配回到 0.000GiB，并已成功重新启动。

## 离线 MLX 就绪权重缓存

首次手动准备一次（保留原模型文件）：

```sh
.venv/bin/mqwen prepare --extra-models /Users/jinjun/ComfyUI-Shared/models
```

默认缓存目录为 `~/Library/Caches/mqwen-infer`，三组模型额外占用约 15GiB 磁盘。这里使用 Safetensors 加 JSON manifest，尚不是单文件 `.mqwen` 容器。准备阶段完成 INT8 → uint32 打包、BF16 尺度/偏置、投影拆分、VAE 卷积布局及精度转换。行尺度和偏置只存一列，加载时恢复广播视图；不重新量化，不改权重精度。每个模型转换后独立释放内存；写入临时目录，完成后才发布缓存。

以后正常启动 WebUI 或 CLI，会优先读取这些缓存：

```sh
.venv/bin/mqwen web --extra-models /Users/jinjun/ComfyUI-Shared/models
```

启动日志显示 `prepared weight cache hit: text_encoder / dit / vae`。启动不会自动创建大文件；缺失缓存时沿用原转换路径。JSON 元数据的 `weight_cache` 分别记录 `hit / miss / disabled / rejected`。手动对比原加载路径可加 `--no-weight-cache`。自定义目录：准备时用 `--output-dir /path/to/cache`，运行时用 `--weight-cache /path/to/cache`。

缓存身份包含源文件路径、大小、修改/变更时间、文件身份和头部 SHA256，以及适配器、MLX、MFLUX 版本。源模型变化或版本不匹配会产生新缓存键；已存在但损坏的缓存被拒绝，重新从原模型加载。当前仍要求原模型文件存在以检查身份，缓存尚不能作为脱离原模型的独立发行工件。旧版本缓存不会自动删除。

准备阶段记录源文件及缓存 payload 的完整 SHA256。启动为节省 I/O，只检查身份、文件大小/修改时间和张量索引，**不会每次重新扫描全部权重做完整哈希校验**。需要完整复核已有缓存时：

```sh
.venv/bin/mqwen prepare --extra-models /Users/jinjun/ComfyUI-Shared/models --verify
```

复核失败时可以删除错误消息对应的派生缓存目录，再重新准备；原模型不受影响。该优化针对启动转换开销，不会直接加快常驻模型的每步去噪，也不代表零毫秒加载或 GPU 零拷贝。

交替测量原权重与准备缓存的引擎启动时间：

```sh
.venv/bin/python scripts/benchmark_startup.py --extra-models /Users/jinjun/ComfyUI-Shared/models --runs 2
```

该脚本不清除操作系统文件缓存，也不计 Python 导入及首次推理图编译；每次关闭后记录实际活跃分配。不要与另一个正在加载/推理的引擎同时运行。

本机验证（M3 Pro 36GiB；两轮按原权重 → 准备缓存交替）：

| 轮次 | 原权重启动 | 准备缓存启动 |
|---|---:|---:|
| 1 | 34.037s | 6.798s |
| 2 | 19.723s | 4.389s |

两轮均命中三组缓存，活跃分配保持约 15.06GiB；关闭后仅余约 8 字节的 MLX 常量分配。以上不是严格冷启动测量，系统文件缓存及其他应用负载会影响结果。首次离线准备约 81.9s；已有有效缓存的再次 `prepare` 检查约 0.009s，完整 `--verify` 约 22.25s。

已用准备缓存运行 512×512、seed 42、标准 8 步，结果与原始基准逐像素一致，运行未导入 Torch。准备缓存的作用是减少启动准备开销；本轮没有把去噪速度变化计为该功能收益。代表性量化 GEMM 的原路径/缓存路径交替微基准未显示稳定回退，执行权重和广播尺度的底层缓冲区尺寸相同。报告见 `reports/startup-benchmark.json`、`reports/prepared-image-verification.json`。

变更权重表示或转换语义时，必须更新 `src/mqwen/prepared.py` 的 `ADAPTER` 标识，以免复用旧工件。

补充同轮完整生图对照：原权重与缓存分别重新建立引擎后，512 标准 8 步去噪为 41.350s / 41.801s，未显示推理加速；这也说明本轮较历史记录更慢的去噪并非缓存路径独有。两张结果均与原基准逐像素一致。首次文本编码受运行时预热影响，未归为缓存收益。完整记录在 `reports/prepared-end-to-end-comparison.json`。


## ConvRot 编译图优化（2026-10-27）

默认路径现在按旋转分组和 dtype 共享预先求值的 Hadamard 常量，替代每次投影各自创建旋转矩阵。矩阵在进入编译图之前建立，给 MLX 复用 Q/K/V、gate/up 中相同输入旋转的机会；不改变 INT8 权重、BF16 运算、采样步数或 sigma 日程。关闭引擎时同时清除这些常量引用。输出 JSON 的 `denoise_optimizations` 记录 `shared_convrot_constants`，无需重新准备权重缓存。

初次 512×512 配对测量曾显示去噪 44.570s → 39.366s，但随后的反转顺序复测**没有确认稳定提速**：同一常驻引擎、seed 42、标准 8 步，原路径 23.761s / 23.838s，共享常量 24.109s / 23.629s（原→共享→共享→原）。四张完整 PNG 逐像素一致。两种路径均值分别约 23.800s / 23.869s，差异在运行波动范围内，不能将早期约 12% 的差值视为已证实收益。保留常量复用以简化重复图构建，不承诺热运行加速。报告为 `reports/denoise-512-repeat.json`；早期记录在 `reports/shared-rotation-benchmark.json`，系统负载及温度未受控，不能跨轮直接比较绝对时间。

可复测完整生成，或只测生产采样器并核对最终 latent 完全相等（不运行 VAE）：

```bash
.venv/bin/python scripts/benchmark_denoise.py --extra-models /Users/jinjun/ComfyUI-Shared/models --runs 2
.venv/bin/python scripts/benchmark_denoise.py --extra-models /Users/jinjun/ComfyUI-Shared/models --width 1024 --height 1024 --denoise-only --report reports/denoise-1024-latents.json
```

脚本只启动一个引擎，逐组交替原路径与共享常量路径；多组时反转顺序。记录全部步骤与排除第一步的均值。第一步可能包含文本 KV 建立和图编译，应与后续步骤分开看。完整图对照可加 `--step-cache` 检查现有近似缓存模式的兼容性；该模式本身仍会改变输出，只要求优化前后的同一种缓存模式一致。测量时先关闭 WebUI 中的推理服务，避免两个模型引擎竞争内存。

清除操作系统文件缓存用于测量冷启动 I/O，不会减少已常驻模型的 GEMM、Attention 或卷积计算量。本次未执行系统级缓存清理。后续算子优化优先测量 ConvRot 与量化矩阵乘的中间读写，再评估实际矩阵尺寸的分块布局、Attention/调制融合及 VAE 内存峰值；单算子微基准加速须通过完整去噪验证。已试验的 SIMD 旋转实现未显示稳定整图收益，未替换默认运算。

1024×1024 的同轮标准 8 步对照：去噪 105.367s → 104.352s；排除第一步的均值 13.150s → 13.061s，差异不足 1%，未显示明显加速。最终 BF16 latent 转为 FP32 后完全相等（不是仅比较解码后的像素）。报告为 `reports/denoise-1024-latents.json`。此次未执行 VAE，因此这些数字不是整图生成时间；此前独立完整流程在 VAE 阶段发生 Metal 内存不足，1024 整图的内存峰值仍需进一步处理。

现有跨步缓存兼容性对照也已完成：512×512、8 步，两条路径均复用第 3、5 步，输出 PNG 完全一致；去噪分别为 17.984s / 17.694s，未显示新的明显加速。报告为 `reports/denoise-512-cache-compat.json`。跨步缓存本身仍是近似模式，默认关闭。


## M3 Pro 专用 INT8 Metal GEMM（2026-10-27）

新后端复用 MLX 0.32.4 的 Steel MMA 和仿射解码逻辑，将投影 GEMM 的块设为 **64×64×32**，并针对本模型每行共用一个量化尺度/偏置的格式简化寻址。INT8 代码、BF16 解码及激活保持原样，不降低位宽、不减少去噪步数。MLX 原来的量化矩阵乘本身已融合解码；本次调整的是其分块和尺度访问，**ConvRot 仍是独立旋转算子**。

默认 `--gemm-backend auto` 仅在本机已验证的 Apple M3 Pro / `applegpu_g15s`、MLX 0.32.4 上启用。启动时检查每层尺度/偏置是否确实按行共享，并对三种投影几何执行编译与参考值核对；失败时使用原 MLX 路径。适用 BF16、256～4096 个图像 token，以及输入/输出维度 4096→4096、4096→12288、12288→4096；其他设备、精度和尺寸继续使用原 MLX 算子。文本编码器、VAE 不启用该内核。启动日志记录实际启用的 DiT 层数（本模型为 224），输出 JSON 的 `gemm_backend` / `gemm_layers` 记录当前尺寸使用的后端及启用层数。无需重新准备权重缓存。

512×512、seed 42、标准 8 步、一个常驻引擎的反转顺序对照：

| 运行顺序 | 去噪耗时 | 与原始标准基准像素比较 |
| --- | ---: | --- |
| 原 MLX | 24.064s | 完全一致 |
| 专用 GEMM | 22.028s | 完全一致 |
| 专用 GEMM | 21.900s | 完全一致 |
| 原 MLX | 23.299s | 完全一致 |

两条路径均值为 23.682s / 21.964s，去噪耗时下降约 **7.3%**；排除第一步后，每步均值为 2.938s / 2.741s。报告为 `reports/tuned-denoise-benchmark.json`。这些是同机配对结果，系统负载及温度未受控，不是所有提示词、芯片或分辨率的性能承诺。

可在服务启动或 CLI 生成时指定 `--gemm-backend mlx`，恢复原 GEMM。WebUI 切换该启动参数需重启网页服务进程；无需更换模型文件。可复测完整 512 生成及 1024 去噪 latent：

```bash
.venv/bin/python scripts/benchmark_denoise.py --optimization gemm --extra-models /Users/jinjun/ComfyUI-Shared/models --runs 2 --report reports/metal-gemm-512.json
.venv/bin/python scripts/benchmark_denoise.py --optimization gemm --extra-models /Users/jinjun/ComfyUI-Shared/models --width 1024 --height 1024 --denoise-only --report reports/metal-gemm-1024-latents.json
```

另有 `scripts/benchmark_convrot.py`，使用实际 DiT 投影权重检查按 token 行/输出通道切分的数值及耗时。它测的是单算子，不能当作整图收益。该轮大多数外层切分未显示稳定改善；底层 64×64×32 分块才在完整 512 去噪中保留了收益。运行任何基准前先关闭网页中的推理服务，避免模型引擎竞争内存。Metal 内核保留 Apple MIT 版权及许可证，见 `THIRD_PARTY.md`。

底层 tile 搜索可运行 `.venv/bin/python scripts/benchmark_gemm_tiles.py`。它使用同一份随包分发的 Metal 源码，比较 18 种 MMA 分块与原 MLX 量化 GEMM，逐项记录输出误差和交替顺序的多轮耗时；只用于研究，不会修改权重或服务配置。

生产后端 1024×1024 的标准 8 步同轮对照也已完成：原 MLX 去噪 103.358s，专用 GEMM 为 98.633s，耗时下降约 **4.6%**；排除第一步后每步均值 12.898s / 12.344s。最终 BF16 latent 完全一致。报告为 `reports/metal-gemm-1024-latents.json`。该对照未运行 VAE，不代表 1024 整图时间，也未解决此前的 VAE 内存峰值问题。

生产路径另已完整生成 512 标准图和跨步缓存图，分别与各自原基准逐像素一致；缓存模式仍复用第 3、5 步。实际取消缓存任务后已恢复执行状态，关闭引擎的 MLX 活跃分配显示 0.000GiB（记录为 18 字节常量）。验证报告为 `reports/metal-gemm-verification.json`，运行未导入 Torch。该项优化不修复 1024 的 VAE 内存峰值。


## 固定 8 步输出的后续实验

在“保持当前 8 步输出与画质”的约束下，继续测试了以下候选。它们没有替换默认路径，原来的 M3 Pro 专用 GEMM 保持启用。

| 实验 | 结果 | 数值核对 |
| --- | --- | --- |
| 扩大 MMA 分块、改变 SIMD-group 排布 | 没有稳定超过现有配置 | 所测矩阵输出完全一致 |
| gate/up 双 GEMM + SwiGLU 寄存器融合 | 最佳单算子仅快约 0.7%～0.9%，未确认整图收益 | 所测矩阵输出完全一致 |
| INT8 N64/K32 tile-major 权重重排 | 约在 ±2% 波动，未确认收益 | 打包可逆、所测输出完全一致 |
| 在寄存器中提前读取下一个 K 块 | 本轮所测候选约慢 2%～11% | 所测输出完全一致 |
| FP16 输入 MMA / FP32 累加，并逐块检查能否精确表示 | 检查版约慢 14%～17%，未采用 | 检查版所测输出完全一致；未检查版有一次出现数值差异且仍更慢 |

报告分别为 `reports/warp-qmm-benchmark.json`、`fused-ffn-benchmark.json`、`tiled-weights-benchmark.json`、`prefetch-qmm-benchmark.json`、`mixed-qmm-benchmark.json`。实验代码位于本地 `reports/experimental_*.py`，属于未启用的研究原型，未随运行时分发。微基准使用实际投影权重和合成 BF16 激活，不能代替完整生图结果。

从模型维度计算，512×512 标准 8 步的主要线性投影约需 114.35 万亿次稠密等效浮点运算（每个乘加按两次操作计），尚未计入 Attention、旋转、归一化和 VAE。将三类代表性投影的微基准耗时按各层调用次数外推，约为 21.9s。这支持“当前主要时间花在矩阵乘本身”的判断；**它不是精确的 GPU 时间分解，也不是硬件理论下限**，缓存、编译图与系统负载会影响外推。汇总见 `reports/strict-eight-step-summary.json`。

目前没有证据支持在当前机器上、不改变 8 步输出的同时，把约 22 秒稳定压到 10 秒级。后续仍可研究更深的 GEMM 实现，但不会将未验证的单算子差值或近似计算当作标准路径的加速收益。当前机器测试期间使用交流电，系统报告低电量模式关闭；未修改系统电源设置。


## 输出块调度与 MPP 对照（2026-10-27）

增加 `scripts/gemm_walk.py` 和 `scripts/benchmark_gemm_walk.py`，测试一维行/列优先、按 M 分组、局部及矩形 Morton 等 10 种输出块遍历方式。它们复用现有 64×64×32 GEMM，不改变单元素内部 K 维乘加顺序。矩形、非整块和填充区域有独立覆盖检查；GPU 边界测试及真实权重微基准的输出完全一致。该工具没有接入默认运行时。

```bash
.venv/bin/python scripts/benchmark_gemm_walk.py --tokens 1024 4096
```

先关闭 WebUI 的推理引擎再运行基准。报告默认写入 `reports/gemm-walk-benchmark.json`。它使用实际投影权重和合成 BF16 激活，记录正反顺序多轮耗时；不代表完整生成时间。

512×512、seed 42、标准 8 步的完整生图反转顺序对照：现有调度 22.113s / 21.997s，按投影几何选择 Morton 的候选 21.966s / 21.955s。均值 22.055s / 21.961s，排除第一步的均值 2.7479s / 2.7434s，差异约 0.2%。四张 PNG 均与原始标准基准逐像素一致；该差异不足以确认稳定提速，因此未改默认调度。报告为 `reports/walk-denoise-benchmark.json`。

另在本机 macOS 26.3 的 M3 Pro 上验证了 MPP TensorOps 可编译运行。采用静态张量范围、`relaxed_precision=false`，比较 7 组块/线程组设置；输入为 BF16 激活及预先解码的 BF16 权重，FP32 输出再转 BF16。三类实际矩阵的最佳参考值：

| 投影 | 现有 INT8 GEMM | 最佳 MPP BF16 参考 |
| --- | ---: | ---: |
| 4096→4096 | 6.325ms | 6.315ms |
| 4096→12288 | 18.425ms | 18.008ms |
| 12288→4096 | 18.669ms | 18.218ms |

所测 BF16 输出均与当前内核完全一致。这是**扩大权重表示后的稠密参考路径**，不含每次 INT8 解码，也不是完整生成；多一份 FP32 输出缓冲区及转换。它未显示足以支撑替换后端的明显优势，没有接入模型引擎。报告及本地研究原型为 `reports/mpp-reference-benchmark.json`、`reports/experimental_mpp.py`；这些未启用的研究代码未随运行时分发。

此前微基准外推只能说明 GEMM 是主要耗时，不能证明硬件算力已经饱和。完整的 GPU occupancy/执行单元/缓存限制分析仍待硬件计数器验证：本机当前只有 Command Line Tools，找不到 `xctrace`，未安装或修改系统开发工具。


## 矩阵内核的 GPU 时间测量（2026-10-28）

通过独立 Swift/Metal 工具加载当前相同的 64×64×32 INT8 GEMM，使用实际三类投影权重与合成 BF16 激活，读取 `MTLCommandBuffer.gpuStartTime/gpuEndTime`。结果逐字节对比生产 MLX 内核的输出，全部一致。排除首轮预热后，三轮中位数如下（M=1024，对应 512×512 图像 token 数）：

| 投影 | 单次提交 GPU 时间 | 每批 8 次的平均 GPU 时间 |
| --- | ---: | ---: |
| 4096→4096 | 5.999ms | 5.980ms |
| 4096→12288 | 17.822ms | 17.820ms |
| 12288→4096 | 18.238ms | 18.288ms |

单次提交的墙钟时间分别为 6.634 / 18.521 / 18.534ms，批量提交后为 6.063 / 17.908 / 18.327ms。这支持摊薄提交和完成等待的开销，但没有显著缩短矩阵的 GPU 执行时间。这里重复的是共享输入的独立 GEMM，**不是完整去噪图**，不能将差值直接乘以生图调用次数作为预期收益。

另测试 Metal pipeline 的最大线程数提示 128、256、512、1024；实际 dispatch 都保持 128 线程。GPU 时间差在约 0.5% 范围内，没有发现明显收益，未修改生产实现。编译器报告的静态 threadgroup 内存为 10240 bytes；这与最大线程数属性均不能代替实际 occupancy 测量。

本机运行时枚举只暴露 `timestamp/GPUTimestamp`，支持 stage boundary、不支持 dispatch boundary 采样。没有拿到执行单元、占用率或缓存命中率计数器，因此不能据此宣称 GPU 已饱和。当前标准 8 步路径及输出保持不变，本轮没有确认新的整图加速。

原始结果：`reports/gemm-gpu-timing.json`、`reports/gemm-thread-hint.json`；汇总：`reports/gemm-gpu-profile-summary.json`。本地研究工具为 `reports/list_metal_counters.swift`、`reports/prepare_gpu_profile.py`、`reports/profile_gemm_gpu.swift`、`reports/profile_gemm_threads.swift`，未随运行时分发。复现时先停止推理引擎，运行权重导出，再用 `swiftc -O` 编译并执行 Swift 工具（参数为 `reports/gpu-profile-data`）；导出的临时权重可在结束后删除。
