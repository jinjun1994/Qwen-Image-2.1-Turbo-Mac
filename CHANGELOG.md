# Changelog

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
