# Qwen-Image 2.1 Turbo capabilities and model choices

Checked on 2026-10-28. [Chinese details](CAPABILITIES_AND_MODELS.zh-CN.md).

The [official Turbo model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) describes the same 7B checkpoint serving generation and editing in 8 steps, CFG=1. The [official overview](https://qwen.ai/blog?id=qwen-image-2.1) describes transparent outputs, up to 10 references and subject-preserving edits.

MQwen now exposes RGB/RGBA output, ordered references, background-removal instructions, a painted or uploaded grayscale mask, and person/product preservation instructions. The vision tower is loaded from the existing Qwen3-VL file on first edit; the resident language model, DiT and VAE are reused. References are resized proportionally to approximately 512². Editing increases memory and processing time; 10 references with 2K output is not guaranteed to fit every Mac.

Black mask pixels preserve the resized first source exactly through a final composite, including alpha in RGBA mode. White pixels use the generated result; gray pixels blend them. Keep the source and output aspect ratio the same to avoid stretching. Unmasked instruction editing has model-dependent fidelity, with no promise of exact identity, logos or fine details.

RGBA is one transparent image. Automatic decomposition into multiple semantic layers belongs to a separate model such as Qwen-Image-Layered and is not included here. Automatic object masks, prompt rewriting and verification are also not enabled. Experimental approximate cross-step caching is disabled for reference editing.

## Weight alternatives

| Component | Current file | Alternatives |
| --- | --- | --- |
| VAE | BF16, 675,509,688 bytes (~676 MB) | Official Comfy-Org currently lists this version. FP16 also uses 16 bits and does not meaningfully reduce size. |
| Tiny VAE | Not enabled by default | [TAEQI2.1](https://huggingface.co/madebyollin/taeqi2_1) is 30,591,264 bytes (~30.6 MB); Tinier is 8,200,064 bytes (~8.2 MB). They support the same 64-channel, 16×, RGBA latent interface but are distilled approximations. They require a different loader and are not integrated in this change. |
| Turbo DiT | INT8 ConvRot, ~7.26 GB | Official Turbo BF16 is 14,230,280,616 bytes (~14.23 GB). It reduces weight quantization error but uses more memory; speed and visual gains require testing. |
| Qwen3-VL | INT8 ConvRot, ~9.35 GB | Official BF16 is 17,534,334,616 bytes (~17.53 GB). A different encoder architecture or parameter count is not a compatible replacement. |
| Lower-bit formats | INT8 remains the default | [Community Turbo MLX 4/6-bit and GGUF files](https://huggingface.co/abenzerps/Qwen-Image-2.1-Turbo-Quantized) exist. They require compatible loaders and weight layouts, and are not interchangeable by renaming a file. They have not been benchmarked by this project. |

Sizes were checked against repository file metadata: [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/0217efd1d9c9452e84f6448903d6e21bfa171ffd), and TAEQI2.1 revision `5d62f11fee0068b4df7bd699ddab685e32174010`. Model licenses are independent of the MQwen source license.

For the current 36 GB Mac and unchanged image quality, retain INT8 Turbo + INT8 Qwen3-VL + the full BF16 VAE. A smaller VAE affects VAE resource use, not the DiT matrix operations. Tiny VAE and lower-bit models need a separate quality/performance comparison before changing defaults.

## API and CLI

Upload PNG/JPEG/WebP bytes to `POST /api/uploads` with the matching image `Content-Type`. Use the returned 32-character ID in the existing `POST /api/generate` request:

```json
{
  "prompt": "Change the teapot in <image1> to blue. Keep everything else unchanged.",
  "seed": "42", "mode": "resolution", "width": 512, "height": 512,
  "steps": 8, "step_cache": false,
  "references": ["32-character ID from upload"],
  "mask": null, "output_format": "rgba"
}
```

`mask` optionally identifies another uploaded grayscale image. Uploads stay in `.inputs/` under the local output folder; no external service receives them. Limits: 20 MiB per upload, 16 MP and aspect ratio between 1:8 and 8:1. API IDs only resolve this upload folder, not arbitrary local paths.

```bash
mqwen generate --image-paths source.png reference.png \
  --mask-image mask.png --output-format rgba \
  --prompt 'Change only the masked region of <image1> using <image2>.' \
  --width 512 --height 512 --seed 42 --steps 8 --output edited.png
```

Existing `--dit`, `--text-encoder` and `--vae` path overrides remain available. With no reference images, the original text-to-image path is used.
