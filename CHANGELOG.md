# Changelog

## 0.5.0 - 2026-10-28

### Features

- Add native MLX reference editing with up to 10 ordered RGB/RGBA images, reusing the resident language model, DiT and full VAE. Load the vision tower from the existing Qwen3-VL checkpoint on first edit.
- Expose true RGBA PNG output, background-removal instructions, and editable person/product preservation instructions.
- Add painted and uploaded masks with latent blending and a final composite that preserves unmasked source pixels.
- Add bounded local image uploads and CLI/API editing options; keep the original RGB text-to-image defaults.

### Documentation

- Document model capabilities separately from guaranteed pixel preservation and automatic layer decomposition.
- Compare full BF16 weights, tiny VAE alternatives and lower-bit formats without changing the default model precision.

### Validation

- Pass 63 Python tests and 2 browser locale tests; check reference upload, painting and background removal through the WebUI.
- Complete seven real-weight source runs and one frozen-runtime run, including masked multi-reference edits and RGBA reference edits. Verify unmasked pixels, no Torch imports and pixel-identical legacy RGB output.

## 0.4.0 - 2026-10-28

### Features

- Add Random prompt / 随机提示词 to fill one of six supplied presets without submitting a generation job.

- Default to English in the desktop app and WebUI, with a persistent Chinese language option.
- Synchronize desktop menus, toolbar and embedded WebUI language without resetting generation inputs or reloading models.
- Localize model settings, downloads, status and validation errors using a shared translation catalog.
- Add confirmed image deletion from the recent gallery and preview: move the PNG and JSON together to the system Trash, with rollback on failure.

### Validation

- Cover language negotiation, translation completeness, persistence, input preservation and protected prompt data.
- Cover trash success, failure rollback, symlinks, missing sidecars and protection for running jobs.

## 0.3.0 - 2026-10-28

### Features

- Add an Apple Silicon desktop app and DMG with a bundled MLX/Metal runtime.
- Provide WebUI, CLI and asynchronous local generation APIs with resident model controls.
- Support custom model paths and verified, resumable downloads from public sources.
- Add specialized INT8 Metal GEMM on the validated M3 Pro configuration, with an MLX fallback.

### Documentation

- Provide Chinese and English getting-started guides, API usage and measured performance.
- Archive detailed optimization experiments in a separate document.

### Known limitations

- The desktop build is ad-hoc signed and not notarized.
- 512×512 is recommended; recent full 1024×1024 decoding encountered GPU OOM.
