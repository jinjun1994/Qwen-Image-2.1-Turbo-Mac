"""Read a supported Comfy API workflow as data; never execute embedded prompts."""
from __future__ import annotations
import json
import math
from pathlib import Path

class Workflow:
    def __init__(self, path):
        self.path = Path(path)
        self.nodes = json.loads(self.path.read_text())

    def inputs(self, kind):
        matches = [n["inputs"] for n in self.nodes.values() if n["class_type"] == kind]
        if len(matches) != 1:
            raise ValueError(f"Expected one {kind}, found {len(matches)}")
        return matches[0]

    def resolve(self, value, seen=None):
        if not isinstance(value, list):
            return value
        seen = set() if seen is None else set(seen)
        node_id, output = value
        if output != 0 or node_id in seen:
            raise ValueError("Unsupported output or cyclic prompt link")
        seen.add(node_id)
        node = self.nodes[node_id]
        data = node["inputs"]
        kind = node["class_type"]
        if kind == "PreviewAny":
            return self.resolve(data["source"], seen)
        if kind == "ComfySwitchNode":
            if type(data["switch"]) is not bool:
                raise ValueError("Only literal boolean switches are supported")
            return self.resolve(data["on_true" if data["switch"] else "on_false"], seen)
        if kind == "PrimitiveStringMultiline":
            return data["value"]
        raise ValueError(f"Prompt requires an unsupported executor: {kind}")

    def plan(self):
        sigmas = [float(s.strip()) for s in self.inputs("ManualSigmas")["sigmas"].split(",")]
        if len(sigmas) < 2 or not all(math.isfinite(s) for s in sigmas) or sigmas[-1] != 0:
            raise ValueError("Invalid sigma schedule")
        if any(a <= b for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError("Sigmas must strictly decrease")
        clips = [n["inputs"]["clip_name"] for n in self.nodes.values()
                 if n["class_type"] == "CLIPLoader" and n["inputs"]["type"] == "qwen_image"]
        if len(clips) != 1:
            raise ValueError("Expected one Qwen image text encoder")
        latent = self.inputs("EmptyLatentImage")
        return {"workflow": str(self.path), "models": {
                    "dit": self.inputs("UNETLoader")["unet_name"],
                    "text_encoder": clips[0], "vae": self.inputs("VAELoader")["vae_name"]},
                "prompt": self.resolve(self.inputs("TextEncodeQwenImage21")["prompt"]),
                "sigmas": sigmas, "steps": len(sigmas)-1,
                "sampler": self.inputs("KSamplerSelect")["sampler_name"],
                "cfg": self.inputs("CFGGuider")["cfg"],
                "seed": self.inputs("RandomNoise")["noise_seed"],
                "batch_size": latent["batch_size"],
                "resolution_selector": self.inputs("ResolutionSelector"),
                "text_encoding_resolution": self.inputs("TextEncodeQwenImage21")["resolution"]}
