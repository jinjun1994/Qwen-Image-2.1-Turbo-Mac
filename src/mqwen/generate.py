"""Workflow planning for the native MLX pipeline."""
from __future__ import annotations
import math
from pathlib import Path

from .weights import WeightFile
from .workflow import Workflow

FOLDERS = {"dit": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}
TURBO_SIGMAS = [1.0, .978453, .95418, .926626, .89508, .845148, .704534, .414568, 0.0]


def model_paths(plan, roots, overrides):
    resolved = {}
    for role, name in plan["models"].items():
        if overrides.get(role):
            candidates = [Path(overrides[role]).expanduser()]
        else:
            candidates = [Path(root).expanduser() / FOLDERS[role] / name for root in roots]
        existing = [p.resolve() for p in candidates if p.is_file()]
        if not existing:
            raise ValueError(f"Missing {role}: {name}. Searched: {candidates}")
        resolved[role] = str(existing[0])
        WeightFile(existing[0])
    return resolved


def output_size(selector, width=None, height=None):
    if (width is None) != (height is None):
        raise ValueError("Provide both --width and --height")
    if width is None:
        # Standalone sizing policy, not execution of a third-party custom node.
        ratio = selector["aspect_ratio"].split()[0]
        a, b = (int(x) for x in ratio.split(":"))
        area = float(selector["megapixels"]) * 1_000_000
        multiple = math.lcm(int(selector["multiple"]), 32)
        width = max(multiple, round(math.sqrt(area*a/b)/multiple)*multiple)
        height = max(multiple, round(math.sqrt(area*b/a)/multiple)*multiple)
    if any(type(n) is not int or n < 256 or n > 2048 or n % 32 for n in (width, height)):
        raise ValueError("Image dimensions must be multiples of 32, between 256 and 2048")
    return width, height


def validate_plan(plan):
    if plan["sampler"] != "euler" or plan["cfg"] != 1 or plan["batch_size"] != 1:
        raise ValueError("This Turbo pipeline requires Euler, CFG=1, batch=1")
    if plan["sigmas"] != TURBO_SIGMAS:
        raise ValueError("This backend requires the exact Turbo 8-step sigma schedule")


def generate(args):
    from .native import generate_native
    return generate_native(args)
