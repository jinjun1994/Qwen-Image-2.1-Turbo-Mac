# Desktop implementation

`MQwen.swift` is an AppKit host with a WKWebView. It owns one bundled worker process, validates readiness using its PID, restricts embedded navigation and file-picker messages to the local origin, and terminates only its own service. A random per-launch control token protects the private shutdown route. HTTP binds only 127.0.0.1.

The worker runs the same `mqwen web` and native MLX inference code as the CLI. Desktop mode adds persistent model settings and explicit SHA-256-verified downloads; it does not change model arithmetic. Models load only on request. App termination and service stop wait for cancellation checkpoints and native engine cleanup.

Build from the repository root with `./scripts/build_desktop.sh`. Use `desktop/mqwen-worker.spec` for the standalone runtime, `icon.swift` for the local icon, and `worker.py --runtime-check` in the frozen bundle to validate MLX and model imports without loading weights. `collect_submodules('mlx')` is necessary because the compiled extension imports Python helpers dynamically. Certifi's bundled CA store supports HTTPS downloads without relying on the build machine's Python installation.

The local DMG is ad-hoc signed and not notarized. A public release still needs a Developer ID identity, notarization and tests on independent Macs. Packaging requires no weights and includes dependency license files. See the root README for installation, API usage, measured performance and limitations.
