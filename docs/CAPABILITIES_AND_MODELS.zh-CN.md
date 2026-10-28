# Qwen-Image 2.1 Turbo：能力与权重选择

核对日期：2026-10-28。区分模型本身的能力、云 API 的能力和本地 MQwen 已验证的功能。

## 功能边界

| 功能 | 模型与 MQwen 接入方式 |
| --- | --- |
| 文生图 | 保留原有标准 8 步、CFG=1、Euler 路径，默认 RGB |
| 透明背景 / RGBA 图层 | 使用原有四通道 VAE，选择 RGBA，并在提示词中明确要求透明背景；保存真实 alpha PNG，不是后期抠图 |
| 原图与多图参考编辑 | 同一 Turbo DiT 权重，使用现有 Qwen3-VL 文件内的视觉编码器；最多 10 张，按顺序用 `<image1>`、`<image2>` 引用 |
| 透明图编辑 / 背景移除 | 可上传 RGBA 原图，选择 RGBA 输出；“移除背景”任务提供可编辑的提示词 |
| 涂绘 / 上传蒙版 | 在第 1 张原图上绘制或上传灰度蒙版，白色修改、黑色保留；逐步潜空间约束，加最终像素合成 |
| 人物 / 商品特征保留 | 依赖参考条件和明确指令。按钮把指令添加到提示词供用户修改；不能保证身份、商标或细节完全一致 |
| 自动多图层分解 | 不包含。单张 RGBA 图与多个语义图层不同；Qwen-Image-Layered 是独立模型，不能冒充本模型原生功能 |
| 自动对象蒙版 / 自动提示词扩写 | 本次不接入；需要额外的生成头、推理与效果验证，不等于手动蒙版 |

依据：[Qwen 官方 Turbo 模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)、[Qwen 官方介绍](https://qwen.ai/blog?id=qwen-image-2.1)、[MFLUX 编辑实现](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/qwen21/variants/edit/qwen_image_21_edit.py)。ComfyUI 节点可暴露更多槽位，本项目采用官方说明的最多 10 张，不把节点槽位数当作模型效果保证。

编辑会增加编码耗时、视觉塔常驻内存和前缀 KV 内存。首次编辑按需加载视觉塔，此后保留到停止引擎。普通文生图不需要视觉塔。参考图在保留宽高比的情况下缩放到约 512²；不保证多参考图与 2K 输出在所有 Mac 上都能同时容纳。跨步近似缓存不用于编辑。

蒙版外精确保留指的是：按输出分辨率缩放后的第 1 张原图，在蒙版灰度值为 0 的像素上保持相同颜色值；RGBA 模式也保留 alpha。软边缘属于混合区域。普通无蒙版编辑只由模型决定保留程度。输入与输出宽高比不同时会发生缩放，建议保持一致。

## 更小或更高质量的模型

| 组件 | 当前文件 / 大小（十进制） | 替代方案及取舍 |
| --- | --- | --- |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` / 675,509,688 B，约 676 MB | 官方 Comfy-Org 当前目录仅提供此版本。FP16 与 BF16 都是 16 bit，转换不会显著缩小 |
| 小 VAE | 保持默认完整 VAE | 作者发布 [TAEQI2.1](https://huggingface.co/madebyollin/taeqi2_1)：30,591,264 B（30.6 MB）；Tinier：8,200,064 B（8.2 MB）。均支持 64 通道、16 倍缩放和 RGBA，但为蒸馏近似，改变细节，不能直接用现有 VAE 加载器替换；本次未集成 |
| Turbo DiT | INT8 ConvRot / 7,256,787,829 B，约 7.26 GB | 官方 Turbo BF16 / 14,230,280,616 B，约 14.23 GB，减少权重量化误差但更占内存，速度需实测；不能保证观感提升 |
| Qwen3-VL | INT8 ConvRot / 9,350,798,360 B，约 9.35 GB | 官方 BF16 / 17,534,334,616 B，约 17.53 GB；同样保留视觉塔。更换为不同尺寸或架构的 Qwen 模型并不兼容 |
| 更低位量化 | 当前保留 INT8 | 社区有 Turbo MLX 4/6 bit 与 GGUF 版本。它们需要对应权重布局和加载器，不能仅改路径/扩展名。4 bit 可能更省内存，速度与画质必须在本机对照；尚未验证，不设为默认 |

文件大小来自发布仓库 API；TAEQI2.1 固定核对版本 `5d62f11fee0068b4df7bd699ddab685e32174010`，Comfy-Org 版本 `0217efd1d9c9452e84f6448903d6e21bfa171ffd`。

推荐当前 36 GB Mac 继续使用 **INT8 Turbo + INT8 Qwen3-VL + 完整 BF16 VAE**，先完成编辑质量验证。把 VAE 从 676 MB 换成 30 MB，主要减少 VAE 本身资源；不会让 DiT 的矩阵计算自动变快。对维持当前画质的要求，小 VAE 与 4 bit 都应作为另行对照的可选项。

来源：[官方 Comfy-Org 权重](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main)、[TAEQI2.1 作者说明](https://huggingface.co/madebyollin/taeqi2_1)、[社区 Turbo 量化发布者说明](https://huggingface.co/abenzerps/Qwen-Image-2.1-Turbo-Quantized)。后者的性能描述不作为本项目的 Mac 实测结论。模型许可独立于 MQwen 源代码许可。

## API 与 CLI

先 `POST /api/uploads`，请求体为 PNG/JPEG/WebP 二进制，`Content-Type` 对应图片类型。返回随机 ID；再向原有 `/api/generate` 增加：

```json
{
  "prompt": "Change the teapot in <image1> to blue. Keep everything else unchanged.",
  "seed": "42", "mode": "resolution", "width": 512, "height": 512,
  "steps": 8, "step_cache": false,
  "references": ["上传接口返回的32位ID"],
  "mask": null,
  "output_format": "rgba"
}
```

`mask` 可填另一张已上传的灰度蒙版 ID。图片留在输出目录下 `.inputs/`，不上传外部服务；图片上限 20 MiB、16 MP，宽高比在 1:8 至 8:1。图片 ID 只解析服务自己的上传目录，API 不接受任意本地路径。

```bash
mqwen generate --image-paths source.png reference.png \
  --mask-image mask.png --output-format rgba \
  --prompt 'Change only the masked region of <image1> using <image2>.' \
  --width 512 --height 512 --seed 42 --steps 8 --output edited.png
```

CLI 继续支持已有 `--dit`、`--text-encoder`、`--vae` 自定义路径。未指定 `--image-paths` 则沿用原有文生图路径。
