"""Comfy safetensors -> MLX components. INT8 codes and row scales are preserved."""
from __future__ import annotations
import json
import re
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .weights import WeightFile


def hadamard(size):
    if size < 4 or size & (size-1) or (size.bit_length()-1) % 2:
        raise ValueError('ConvRot group size must be a power of four')
    h4 = mx.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], dtype=mx.float32)
    h = h4
    while h.shape[0] < size:
        h = (h[:,:,None,None]*h4[None,None,:,:]).transpose(0,2,1,3).reshape(h.shape[0]*4,-1)
    return h / size**0.5


@lru_cache(maxsize=16)
def _rotation_matrix(group_size,dtype):
    # The same constant identity lets MLX's compiled graph share equal rotations
    # across Q/K/V and gate/up projections without changing their arithmetic.
    return hadamard(group_size).astype(dtype)


def prime_rotations(groups=(256,)):
    # Build/evaluate outside mx.compile, rather than retaining traced constants.
    mx.eval(*[_rotation_matrix(group,dtype) for group in groups for dtype in (mx.bfloat16,mx.float32)])


def clear_rotations():
    _rotation_matrix.cache_clear()


def rotate(x, group_size):
    shape = x.shape
    if shape[-1] % group_size:
        raise ValueError('ConvRot group does not divide input dimension')
    return (x.reshape(-1, group_size) @ _rotation_matrix(group_size,x.dtype)).reshape(shape)


class ConvRotLinear(nn.Module):
    @classmethod
    def from_packed(cls, weight, scales, biases, group, bias=None):
        """Hydrate the exact saved execution representation without repacking."""
        if weight.ndim!=2 or weight.dtype!=mx.uint32:
            raise ValueError('Prepared INT8 weight must be rank-2 uint32')
        out_features,in_features=weight.shape[0],weight.shape[1]*4
        shape=(out_features,in_features//64)
        if in_features%64 or scales.shape!=shape or biases.shape!=shape:
            raise ValueError('Prepared quantization geometry mismatch')
        if scales.dtype!=mx.bfloat16 or biases.dtype!=mx.bfloat16:
            raise ValueError('Prepared quantization scales must be BF16')
        if group and in_features%group:
            raise ValueError('Prepared rotation does not divide input')
        if bias is not None and bias.shape!=(out_features,):
            raise ValueError('Prepared projection bias mismatch')
        layer=cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.group=group
        layer.group_size=64
        layer.in_features=in_features
        layer.out_features=out_features
        layer.weight,layer.scales,layer.biases=weight,scales,biases
        layer.bias=bias
        return layer

    def __init__(self, codes, scale, config, bias=None):
        super().__init__()
        if config.get('format') != 'int8_tensorwise':
            raise ValueError(f'Unsupported quantization {config}')
        self.group = config.get('convrot_groupsize', 256) if config.get('convrot') else 0
        self.in_features = codes.shape[1]
        self.out_features = codes.shape[0]
        self.group_size = 64
        if self.in_features % 64:
            raise ValueError('Quantized input dimension must be divisible by 64')
        # Map signed [-128,127] to affine uint8 codes without requantization.
        unsigned = (codes.astype(mx.int32)+128).astype(mx.uint32).reshape(self.out_features,-1,4)
        self.weight = mx.sum(unsigned << mx.array([0,8,16,24], dtype=mx.uint32), axis=-1).astype(mx.uint32)
        self.scales = mx.broadcast_to(scale.reshape(-1,1), (self.out_features,self.in_features//64)).astype(mx.bfloat16)
        self.biases = (-128*self.scales).astype(mx.bfloat16)
        self.bias = bias
        mx.eval(self.weight, self.scales, self.biases)

    def __call__(self, x):
        if self.group:
            x = rotate(x, self.group)
        if getattr(self,'_tuned_gemm',False):
            from .metal_gemm import matmul,supported
            if supported(self,x):
                y=matmul(x,self)
                return y if self.bias is None else y+self.bias
        y = mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                transpose=True, group_size=64, bits=8)
        return y if self.bias is None else y+self.bias


class ConvRotEmbedding(nn.Module):
    def __init__(self, codes, scale, config):
        super().__init__()
        self.weight = codes
        self.scale = scale
        self.group = config.get('convrot_groupsize',256) if config.get('convrot') else 0
        mx.eval(self.weight, self.scale)

    def __call__(self, ids):
        x = self.weight[ids].astype(mx.float32)*self.scale[ids]
        return (rotate(x,self.group) if self.group else x).astype(mx.bfloat16)


def get_path(module, path):
    for part in path.split('.'):
        module = module[int(part)] if part.isdigit() else getattr(module,part)
    return module


def set_path(module, path, value):
    parts = path.split('.')
    owner = get_path(module,'.'.join(parts[:-1])) if len(parts)>1 else module
    if parts[-1].isdigit():
        owner[int(parts[-1])] = value
    else:
        setattr(owner,parts[-1],value)


def load_module(module, path, key_transform=lambda k:k, *, vae=False, log=print, weight_transform=None):
    index = WeightFile(path)
    arrays = mx.load(str(path))
    expected = dict(tree_flatten(module.parameters()))
    supplied = set()
    count = 0
    with index:
        for source in index.tensors:
            if not source.endswith(('.weight','.bias','.gamma')):
                continue
            target = key_transform(source)
            if target is None:
                continue
            targets = [target]
            split = target.endswith('.gate_up.weight')
            if split:
                targets = [target.replace('.gate_up.weight','.gate_layer.weight'),
                           target.replace('.gate_up.weight','.proj.weight')]
            if any(k not in expected for k in targets):
                raise ValueError(f'Unexpected mapped weight {source} -> {targets}')
            weight = arrays.pop(source)
            if weight_transform is not None:weight=weight_transform(target,weight)
            pieces = mx.split(weight,2,axis=0) if split else [weight]
            quant_key = source.removesuffix('.weight')+'.comfy_quant'
            config = None
            if quant_key in index.tensors:
                view = index.view(quant_key)
                try: config = json.loads(bytes(view))
                finally: view.release()
            scale_key = source.removesuffix('.weight')+'.weight_scale'
            scale = arrays.pop(scale_key) if config is not None else None
            scales = mx.split(scale,2,axis=0) if split else [scale]
            for name, value, row_scale in zip(targets,pieces,scales):
                if name in supplied:
                    raise ValueError(f'Duplicate mapped key {name}')
                if tuple(value.shape) != tuple(expected[name].shape) and not vae:
                    raise ValueError(f'Wrong weight shape {name}: {value.shape} != {expected[name].shape}')
                if config is not None:
                    if value.dtype != mx.int8:
                        raise ValueError(f'Quantized weight {source} must be signed INT8')
                    parent = name.removesuffix('.weight')
                    layer = ConvRotEmbedding(value,row_scale,config) if parent=='embed_tokens' else ConvRotLinear(value,row_scale,config)
                    set_path(module,parent,layer)
                else:
                    if vae:
                        if value.ndim==5:
                            if value.shape[2] != 1:
                                raise ValueError(f'Temporal kernel cannot run on image path: {source}')
                            value = value[:,:,0]
                        if value.ndim==4 and name.endswith('.weight'):
                            value = value.transpose(0,2,3,1)
                        if name.endswith('.gamma'):
                            value = value.reshape(-1)
                        value = value.astype(mx.float32)
                    else:
                        value = value.astype(mx.bfloat16)
                    if tuple(value.shape) != tuple(expected[name].shape):
                        raise ValueError(f'Wrong converted shape {name}: {value.shape} != {expected[name].shape}')
                    mx.eval(value)
                    set_path(module,name,value)
                supplied.add(name)
            count += 1
            if count % 48 == 0:
                log(f'[mqwen] loaded {count} tensors',flush=True)
        missing = {k for k in set(expected)-supplied if k not in ('rotary_emb.inv_freq','rotary_pos_emb.inv_freq') and k != 'time_text_embed.time_proj.freqs' and not k.startswith(('pos_embed.cos_tables.','pos_embed.sin_tables.'))}
        if missing:
            raise ValueError(f'Missing model parameters: {sorted(missing)[:20]}')
    return module


def text_key(key):
    if key.startswith(('visual.','model.visual.','lm_head.','model.lm_head.')):
        return None
    return key.removeprefix('model.language_model.').removeprefix('model.')


def dit_key(key):
    return key.replace('modulation.1.','modulation.layers.1.')


def vae_key(key):
    if '.time_conv.' in key:
        return None  # Video-only first-frame branches are not used for images.
    if key.startswith('conv1.'):
        return key.replace('conv1.','quant_conv.',1)
    if key.startswith('conv2.'):
        return key.replace('conv2.','post_quant_conv.',1)
    key = re.sub(r'^(encoder|decoder)\.conv1\.',r'\1.conv_in.',key)
    key = key.replace('.head.0.','.norm_out.').replace('.head.2.','.conv_out.')
    key = key.replace('.middle.0.','.mid_block.resnets.0.').replace('.middle.2.','.mid_block.resnets.1.')
    key = key.replace('.middle.1.','.mid_block.attentions.0.')
    key = re.sub(r'encoder.downsamples.(\d+).downsamples.([01]).',r'encoder.down_blocks.\1.resnets.\2.',key)
    key = re.sub(r'encoder.downsamples.(\d+).downsamples.2.',r'encoder.down_blocks.\1.downsampler.',key)
    key = re.sub(r'decoder.upsamples.(\d+).upsamples.([012]).',r'decoder.up_blocks.\1.resnets.\2.',key)
    key = re.sub(r'decoder.upsamples.(\d+).upsamples.3.',r'decoder.up_blocks.\1.upsampler.',key)
    for old,new in [('residual.0.','norm1.'),('residual.2.','conv1.'),('residual.3.','norm2.'),('residual.6.','conv2.'),('shortcut.','conv_shortcut.')]:
        key = key.replace(old,new)
    return key
