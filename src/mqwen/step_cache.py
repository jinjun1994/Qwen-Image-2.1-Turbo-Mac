# Copyright 2026 The Qwen Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
# Model execution adapted from MFLUX Qwen21Transformer; residual cache added locally.
"""Opt-in approximate residual reuse for the eight-step Turbo schedule.

The first two and last two steps run in full; consecutive reuse is forbidden.
Each image starts with an empty cache. This changes the image, not the sigma
schedule. The original uncached path remains the default.
"""
import math
import mlx.core as mx
import mlx.nn as nn


def can_reuse(step, steps, change, has_cache, previous_skipped, threshold):
    return (has_cache and 2 <= step < steps-2 and not previous_skipped
            and math.isfinite(change) and change < threshold)


class CachedImageStep:
    def __init__(self, model, threshold=.08, steps=8):
        self.model = model
        self.threshold = threshold
        self.steps = steps
        self.head = mx.compile(self._head)
        self.body = mx.compile(self._body)
        self.tail = mx.compile(self._tail)
        self.reset()

    def reset(self):
        self.index = 0
        self.signal = self.residual = None
        self.skipped = []
        self.changes = []
        self.previous_skipped = False

    def stats(self):
        return dict(enabled=True, method='first-block residual reuse',
                    threshold=self.threshold, reused_steps=list(self.skipped),
                    relative_changes=list(self.changes),
                    full_steps=self.index-len(self.skipped))

    def _head(self, x, t, kvs, cos, sin):
        model = self.model
        temb = model.time_text_embed(mx.concatenate([t, mx.zeros((1,), dtype=t.dtype)]))
        mod1, mod2 = mx.split(model.modulation(temb), 2, axis=-1)
        mod1 = mx.broadcast_to(mod1[0][None, None, :], (1, x.shape[1], mod1.shape[-1]))
        mod2 = mx.broadcast_to(mod2[0][None, None, :], (1, x.shape[1], mod2.shape[-1]))
        x = model.img_in(x)
        x = model.transformer_blocks[0].image_forward(x, *kvs[0], mod1, mod2, cos, sin)
        return x, mod1, mod2, temb

    def _body(self, x, mod1, mod2, kvs, cos, sin):
        for block, kv in zip(self.model.transformer_blocks[1:], kvs[1:]):
            x = block.image_forward(x, *kv, mod1, mod2, cos, sin)
        return x

    def _tail(self, x, temb):
        scale = self.model.norm_out.linear(nn.silu(temb))
        scale = mx.broadcast_to(scale[0][None, None, :], (1, x.shape[1], scale.shape[-1]))
        return self.model.proj_out(self.model.norm_out(x, scale))

    def __call__(self, x, t, kvs, cos, sin):
        first, mod1, mod2, temb = self.head(x, t, kvs, cos, sin)
        change = (float(mx.mean(mx.abs(first-self.signal)) /
                        (mx.mean(mx.abs(first))+1e-6)) if self.signal is not None else 1.)
        skip = can_reuse(self.index, self.steps, change, self.residual is not None,
                         self.previous_skipped, self.threshold)
        self.changes.append(change)
        if skip:
            result = first+self.residual
            self.skipped.append(self.index+1)
        else:
            result = self.body(first, mod1, mod2, kvs, cos, sin)
            self.residual = result-first
            self.signal = first
        self.previous_skipped = skip
        self.index += 1
        # Re-evaluate the output modulation at the current timestep even on reuse.
        return self.tail(result, temb)
