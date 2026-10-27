from __future__ import annotations
import argparse
import json
from pathlib import Path
from .weights import WeightFile
from .workflow import Workflow

FOLDERS = {"dit": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}

def audit(workflow, models, overrides, extra_models=()):
    plan = Workflow(workflow).plan()
    result = {"plan": plan, "models": {}, "issues": [],
              "inference_ready": False, "backend": "mlx-metal", "audit_only": True}
    total = 0
    for role, filename in plan["models"].items():
        candidates = [Path(root) / FOLDERS[role] / filename for root in [models, *extra_models]]
        path = Path(overrides[role]) if overrides.get(role) else next((p for p in candidates if p.is_file()), candidates[0])
        if not path.is_file():
            result["issues"].append(f"Missing {role}: {path}")
            continue
        weights = WeightFile(path)
        result["models"][role] = weights.summary()
        total += weights.size
        if path.name != filename:
            result["issues"].append(f"Explicit {role} override differs from workflow; compatibility and checkpoint identity unverified")
        if role == "dit":
            quant = [k for k in weights.tensors if k.endswith(".comfy_quant")]
            result["models"][role]["comfy_quant_layers"] = len(quant)
            if quant:
                with weights:
                    view = weights.view(quant[0])
                    try:
                        result["models"][role]["quant_descriptor_example"] = bytes(view).decode("utf-8", errors="replace")
                    finally:
                        view.release()
    result["weight_file_bytes"] = total
    result["memory_note"] = "Weight bytes only; excludes activations, KV, workspaces and OS reserve. mmap does not guarantee residency."
    return result

def main():
    parser = argparse.ArgumentParser(description="Qwen Image 2.1 Mac runtime tools")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate safetensors headers without loading payloads")
    inspect.add_argument("file")
    check = sub.add_parser("doctor", help="Audit workflow and model compatibility")
    check.add_argument("--workflow", required=True)
    check.add_argument("--models", required=True)
    check.add_argument("--extra-models", action="append", default=[])
    for role in FOLDERS:
        check.add_argument("--" + role.replace("_", "-"))
    check.add_argument("--output")
    gen = sub.add_parser("generate", help="Generate a PNG with the native MLX/Metal backend")
    gen.add_argument("--workflow", default=str(Path(__file__).resolve().parent / "assets/turbo_t2i.json"))
    gen.add_argument("--models", default=str(Path.home() / "Documents/ComfyUI2/models"))
    gen.add_argument("--extra-models", action="append", default=[])
    gen.add_argument("--tokenizer", default=str(Path(__file__).resolve().parent / "assets/tokenizer"))
    gen.add_argument("--prompt")
    gen.add_argument("--seed", type=int)
    gen.add_argument("--steps", type=int, choices=[4,6,8], default=8)
    gen.add_argument("--step-cache", action='store_true', help='Experimental approximate residual reuse; requires 8 steps')
    gen.add_argument("--wired-memory", choices=['auto','off'], default='auto', help='Bounded Metal physical residency (default: auto)')
    gen.add_argument("--width", type=int)
    gen.add_argument("--height", type=int)
    gen.add_argument("--output", required=True)
    for role in FOLDERS:
        gen.add_argument("--" + role.replace("_", "-"))
    web = sub.add_parser("web", help="Start the local WebUI")
    web.add_argument("--desktop",action="store_true",help="Enable persistent model settings and downloads")
    web.add_argument("--config-dir",default=None)
    web.add_argument("--no-auto-start",dest="auto_start",action="store_false",default=True)
    web.add_argument("--port", type=int, default=7860)
    web.add_argument("--wired-memory", choices=['auto','off'], default='auto', help='Bounded Metal physical residency (default: auto)')
    web.add_argument("--output-dir", default="outputs")
    web.add_argument("--workflow", default=str(Path(__file__).resolve().parent / "assets/turbo_t2i.json"))
    web.add_argument("--tokenizer", default=str(Path(__file__).resolve().parent / "assets/tokenizer"))
    web.add_argument("--models", default=str(Path.home() / "Documents/ComfyUI2/models"))
    web.add_argument("--extra-models", action="append", default=[])
    for role in FOLDERS:
        web.add_argument("--" + role.replace("_", "-"))
    default_cache=str(Path.home()/"Library/Caches/mqwen-infer")
    for command in (gen,web):
        command.add_argument('--gemm-backend',choices=['auto','mlx'],default='auto',help='Auto: tuned INT8 Metal GEMM on validated M3 Pro; mlx: original kernels')
        command.add_argument('--weight-cache',default=default_cache,help='Read-only MLX-ready weight cache directory')
        command.add_argument('--no-weight-cache',dest='weight_cache',action='store_const',const=None,help='Always convert original weights at startup')
    prep=sub.add_parser('prepare',help='Build offline MLX-ready weights; original models are preserved')
    prep.add_argument('--workflow',default=gen.get_default('workflow'))
    prep.add_argument('--models',default=gen.get_default('models'))
    prep.add_argument('--extra-models',action='append',default=[])
    prep.add_argument('--output-dir',default=default_cache)
    prep.add_argument('--verify',action='store_true',help='Rehash existing source/cache payloads instead of only checking metadata')
    for role in FOLDERS:
        prep.add_argument('--'+role.replace('_','-'))
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            from .prepared import prepare
            print(json.dumps(prepare(args),ensure_ascii=False,indent=2))
            return 0
        if args.command == "web":
            from .web import serve
            serve(args)
            return 0
        if args.command == "generate":
            from .generate import generate
            generate(args)
            return 0
        if args.command == "inspect":
            result = WeightFile(args.file).summary()
        else:
            result = audit(args.workflow, args.models,
                           {r: getattr(args, r) for r in FOLDERS if getattr(args, r)}, args.extra_models)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if getattr(args, "output", None):
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered + "\n")
        print(rendered)
        return 2 if result.get("issues") else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"mqwen: {exc}\n")

if __name__ == "__main__":
    raise SystemExit(main())
