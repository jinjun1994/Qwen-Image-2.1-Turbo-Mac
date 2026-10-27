# MQwen · Qwen-Image 2.1 Turbo for Mac

[中文](README.md) · **English**

A local image studio for Apple Silicon, with a desktop app, WebUI, CLI and image-generation API. Text encoding, DiT denoising and VAE decoding run on **MLX / Metal**, using existing ComfyUI INT8 ConvRot weights. Inference does not require ComfyUI or PyTorch.

## Download and install

Download `MQwen-0.3.0-arm64.dmg` from [Releases](https://github.com/jinjun1994/Qwen-Image-2.1-Turbo-Mac/releases), open it and drag `MQwen.app` into Applications. The desktop app bundles Python and runtime dependencies, **but not model weights**.

- **Requirements:** Apple Silicon and macOS 26+. Tested on M3 Pro / 36GB / macOS 26.3. Other configurations have not been tested; weight size is not the total memory requirement.
- **Signing:** this is an ad-hoc signed test build, without Apple Developer ID signing or notarization. Gatekeeper may block it on other Macs.
- **Privacy:** prompts and images are processed locally. Network access is used for model downloads you explicitly start. No telemetry or cloud inference.

## Generate your first image

1. Open MQwen. The top bar controls the **WebUI / API HTTP service**, at `http://127.0.0.1:7861` by default. Change the port if needed.
2. Open “模型与 API” (Models & API), choose the Turbo DiT, text encoder and VAE files, then save their paths.
3. Click “启动推理服务” (Start inference service) and wait for the models to load.
4. Enter a prompt and set total pixels or resolution, aspect ratio and seed. Start with **512×512, standard 8 steps, cross-step cache off**.
5. Generate, preview, cancel, browse history or save the PNG. Images and parameter JSON files are stored in `~/Pictures/MQwen/`.

Models remain resident until the inference service is stopped. The page's Stop inference service button releases models while keeping HTTP available. The desktop's Stop WebUI / API button stops HTTP and releases models. Quitting the app stops its own service; shutdown during generation waits for the current GPU operation to finish.

Settings: `~/Library/Application Support/MQwen/settings.json`. Logs: `server.log` in the same directory. The current desktop interface uses Chinese labels.

## Model paths and downloads

Select existing ComfyUI files directly; originals are neither copied nor modified. The current adapter supports this combination, **not arbitrary GGUF, FP8, INT4 or other models**:

| Component | Filename |
| --- | --- |
| Turbo DiT | `qwen_image_2.1_turbo_int8_convrot.safetensors` |
| Text encoder | `qwen3vl_8b_int8_convrot.safetensors` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |

The model download panel supports:

- **Hugging Face:** [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), with a pinned revision and SHA-256 hashes.
- **ModelScope / other public sources:** supply a direct public HTTPS file URL and the publisher's SHA-256. No unverified mirror presets or sign-in management are included.

The three files total approximately **17.28GB**, plus space for converted caches and outputs. Downloads are sequential and support progress, cancellation, resume and SHA-256 verification. Existing models are never overwritten. After completion, choose Use downloaded model, save the path and start inference. Stop inference before changing model paths. Model licenses are separate from this project's MIT license.

## Performance

**M3 Pro / 36GB, macOS 26.3, AC power, 512×512, standard 8 steps, approximate cache disabled.** Models are resident; the bundled teapot prompt and seed 42 are used.

| Pipeline | End-to-end image generation |
| --- | ---: |
| ComfyUI — user-reported, not a controlled comparison | About 90s |
| MQwen desktop — first request after engine loading | 24.431s |
| MQwen desktop — warm requests | 23.669s / 23.486s |

**Measured at about 24 seconds per image on AC power**, with a warm median of **23.578s**. Compared with the user-reported 90s, this is roughly **3.8× faster, or 74% less time**. Timings include text encoding, denoising, VAE decoding and saving, but exclude engine startup. Warm requests reuse the same prompt and shape. All three outputs match this project's standard reference pixel for pixel.

These are measured results, not a fixed latency guarantee or a controlled ComfyUI A/B benchmark. Prompt length, chip, temperature and background load affect speed; identical seeds across frameworks do not imply identical images. See the [benchmark record](docs/benchmarks/desktop-ac.json) and [optimization history, in Chinese](docs/PERFORMANCE_NOTES.zh-CN.md).

## Run from source

Requires Apple Silicon, Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/jinjun1994/Qwen-Image-2.1-Turbo-Mac.git
cd Qwen-Image-2.1-Turbo-Mac
./scripts/setup.sh

# WebUI with model settings and downloads; start inference manually
.venv/bin/mqwen web --desktop --no-auto-start
```

Open `http://127.0.0.1:7860`. Press Ctrl+C to stop. The CLI saves to `outputs/` in the current directory by default; change it with `--output-dir`.

Place the model files under `~/MQwen/models/{diffusion_models,text_encoders,vae}/` with the filenames above, then generate:

```sh
.venv/bin/mqwen generate \
  --models "$HOME/MQwen/models" \
  --prompt 'A small red ceramic teapot on a wooden table, morning sunlight.' \
  --seed 42 --width 512 --height 512 --steps 8 \
  --output outputs/teapot.png
```

Use `--dit`, `--text-encoder` and `--vae` for individual files, or `--extra-models` for additional search directories. Models are not downloaded automatically. The output PNG and matching JSON must not already exist. See `mqwen generate --help` and `mqwen web --help`.

## Local image-generation API

The desktop defaults to port **7861**; CLI WebUI defaults to **7860**. Both bind only to `127.0.0.1` and share a single-job resident engine. Public hosting and OpenAI protocol compatibility are not provided.

```sh
curl http://127.0.0.1:7861/api/engine/start \
  -H 'Content-Type: application/json' -d '{}'
# GET /api/state and wait for engine_state == ready

curl http://127.0.0.1:7861/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red ceramic teapot","mode":"resolution","width":512,"height":512,"seed":"42","steps":8,"step_cache":false}'

curl http://127.0.0.1:7861/api/state
# Find the returned job id; use image_url / metadata_url after completion
curl http://127.0.0.1:7861/image/JOB_ID -o result.png
```

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health`, `/api/state` | HTTP identity, engine state, job progress and results |
| `POST /api/generate`, `/api/cancel` | Create / cancel a job; 409 when busy or not ready, 400 for invalid parameters |
| `POST /api/engine/start`, `/api/engine/stop` | Load / release resident models; JSON body `{}` |
| `GET /image/{id}`, `/metadata/{id}` | Completed job's PNG / parameters |
| `GET/POST /api/settings` | Desktop-mode model settings; inference must be stopped before writing |
| `GET /api/downloads` | Model catalog and download status |
| `POST /api/downloads/start`, `/api/downloads/cancel` | Start / cancel downloads; start takes `role`, `source`, plus `url` and `sha256` for custom sources |

All POST requests use `application/json`. Send `seed` as a decimal string. For total-pixel sizing, use `mode:"pixels"` with `megapixels:0.25, ratio:"1:1"` instead of width and height.

## Limitations

- 512×512 is currently recommended. Recent full 1024×1024 VAE decoding encountered GPU OOM; 2048 has not been tested on hardware.
- 6/4-step sampling and cross-step caching are optional experiments that change output. Standard 8 steps and cross-step cache off are the defaults.
- Image editing, LoRA, RGBA and arbitrary model architectures are not supported. Layerwise comparisons with official BF16 weights and broad quality evaluation are not complete.
- Residency does not lock every physical page. Zero allocation, GPU zero-copy loading and ICB replay are not promised.

## Development and licensing

```sh
.venv/bin/python -m unittest discover -s tests -v
uv pip install --python .venv/bin/python pyinstaller==6.22.3
./scripts/build_desktop.sh
```

Desktop builds require Command Line Tools, Swift, `hdiutil` and `iconutil`, producing an app, DMG and SHA-256. See [desktop/README.md](desktop/README.md). Core code is in `src/mqwen/`; the desktop shell uses AppKit / WKWebView, with MFLUX 0.23.0 model components and MLX 0.32.4.

Project code is licensed under [MIT](LICENSE). Third-party derived files, tokenizer assets and weights retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). This is not an official Qwen, Apple or MFLUX product and does not distribute model weights. When reporting issues, include chip, memory, OS, parameters and logs, and remove private prompts and local paths.
