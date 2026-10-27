"""Native MLX/Metal text-to-image using local weights and MFLUX model components."""
from __future__ import annotations
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

# Keep optional backends disabled; tokenization uses the local tokenizers library.
os.environ['USE_TORCH'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import mlx.core as mx
import numpy as np
from PIL import Image, PngImagePlugin
from mflux.models.qwen21.model.qwen21_text_encoder.language_model import LanguageModel
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from .generate import model_paths, output_size, validate_plan
from .workflow import Workflow
from .native_weights import load_module, text_key, dit_key, vae_key,prime_rotations,clear_rotations,ConvRotLinear,ConvRotEmbedding
from .tokenizer import QwenTokenizer, PROMPT_TEMPLATE, SYSTEM_PREFIX
from .memory import residency_budget
from .prepared import load_or_convert,DEFAULT_CACHE,ADAPTER

TEXT_CONFIG = dict(vocab_size=151936,hidden_size=4096,num_hidden_layers=36,
                   num_attention_heads=32,num_key_value_heads=8,intermediate_size=12288,
                   max_position_embeddings=262144,rope_theta=5000000.0,rms_norm_eps=1e-6,
                   head_dim=128,attention_bias=False,rope_scaling={'mrope_section':[24,20,20]})


def denoise_latents(transformer,embeds,width,height,seed,schedule,*,log=print,check_cancel=lambda:None):
    """Run the production sampler; also usable for decoder-independent benchmarks."""
    # MLX and PyTorch use different RNG algorithms. The seed is native MLX,
    # not a promise of pixel identity with ComfyUI's CPU noise generator.
    latents = mx.random.normal((1,(height//16)*(width//16),64),
                               key=(mx.random.fold_in(mx.random.key(seed & 0xffffffff),seed >> 32) if seed >> 32 else mx.random.key(seed))).astype(mx.bfloat16)
    sigmas = mx.array(schedule,dtype=mx.float32)
    config = SimpleNamespace(width=width,height=height,scheduler=SimpleNamespace(sigmas=sigmas))
    for step in range(len(schedule)-1):
        check_cancel()
        step_start = time.perf_counter()
        prediction = transformer(t=step,config=config,hidden_states=latents,
                                 encoder_hidden_states=embeds)
        latents = latents + prediction.astype(latents.dtype)*(sigmas[step+1]-sigmas[step]).astype(latents.dtype)
        mx.eval(latents)
        log(f'[mqwen] denoise {step+1}/{len(schedule)-1}: {time.perf_counter()-step_start:.3f}s',flush=True)
    return latents


def _generate(args, engine, cancelled=lambda: False):
    print = engine.log
    start = time.perf_counter()
    plan = Workflow(args.workflow).plan()
    validate_plan(plan)
    prompt = args.prompt if args.prompt is not None else plan['prompt']
    if not isinstance(prompt,str) or not prompt.strip():
        raise ValueError('Prompt must be non-empty')
    seed = plan['seed'] if args.seed is None else args.seed
    if not 0 <= seed < 2**64:
        raise ValueError('Seed must be an unsigned 64-bit integer')
    steps = getattr(args,'steps',8)
    if type(steps) is not int or steps not in (4,6,8):
        raise ValueError('Supported Turbo step counts: 4, 6, 8')
    schedule = plan['sigmas'] if steps == 8 else np.interp(np.linspace(0,8,steps+1),np.arange(9),plan['sigmas']).tolist()
    width,height = output_size(plan['resolution_selector'],args.width,args.height)
    if plan['models'] != engine.model_names:
        raise ValueError('Workflow models changed; restart the resident engine')
    tuned_for_shape=bool(engine.gemm_layers and 256<=(width//16)*(height//16)<=4096)
    paths = engine.paths
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != '.png' or output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Choose a new .png output path (image and sidecar must not exist)')
    if not mx.metal.is_available():
        raise ValueError('MLX Metal device is unavailable')
    output.parent.mkdir(parents=True,exist_ok=True)
    # Bound the reusable allocator cache, independently of live model weights.
    mx.set_cache_limit(512*1024**2)
    timings = {}
    def mark(stage,begin):
        mx.synchronize()
        timings[stage] = round(time.perf_counter()-begin,3)
        print(f'[mqwen] {stage}: {timings[stage]:.3f}s',flush=True)
    def check_cancel():
        if cancelled():
            raise GenerationCancelled('Generation cancelled')
    print(f'[mqwen] MLX {mx.__version__} / Metal; {width}x{height}; seed={seed}; {steps} steps',flush=True)
    begin = time.perf_counter()
    tokenizer = engine.tokenizer
    check_cancel()
    template = PROMPT_TEMPLATE.format(prompt)
    token_ids = tokenizer.encode(template)
    if len(token_ids)>2048:
        raise ValueError('Prompt exceeds 2048 tokens; shorten it explicitly')
    drop = len(tokenizer.encode(SYSTEM_PREFIX))
    cache_hit = prompt in engine.prompts
    if cache_hit:
        embeds = engine.prompts.pop(prompt)
        engine.prompts[prompt] = embeds
        print('[mqwen] prompt cache hit',flush=True)
    else:
        ids = mx.array([token_ids],dtype=mx.int32)
        positions = mx.broadcast_to(mx.arange(len(token_ids))[None,None,:],(3,1,len(token_ids)))
        embeds = engine.text(engine.text.embed_tokens(ids),positions)[:,drop:]
        mx.eval(embeds)
        if not bool(mx.all(mx.isfinite(embeds)).item()):
            raise RuntimeError('Non-finite text features')
        engine.prompts[prompt] = embeds
        while len(engine.prompts)>4:
            engine.prompts.pop(next(iter(engine.prompts)))
    mark('text_encode',begin)
    begin = time.perf_counter()
    check_cancel()
    transformer = engine.transformer
    print('[mqwen] resident DiT reused',flush=True)
    mark('dit_load',begin)
    begin = time.perf_counter()
    latents = denoise_latents(engine.transformer,embeds,width,height,seed,schedule,
                              log=print,check_cancel=check_cancel)
    if engine._active_step_cache is not None:
        reused = engine._active_step_cache.stats()['reused_steps']
        print(f'[mqwen] residual cache reused steps: {reused}; threshold=0.08',flush=True)
    if not bool(mx.all(mx.isfinite(latents)).item()):
        raise RuntimeError('Non-finite denoised latents')
    mark('denoise',begin)
    check_cancel()
    begin = time.perf_counter()
    vae = engine.vae
    nchw = latents.reshape(1,height//16,width//16,64).transpose(0,3,1,2).astype(mx.float32)
    decoded = vae.decode(nchw)
    pixels = mx.clip(decoded/2+0.5,0,1).transpose(0,2,3,1)
    mx.eval(pixels)
    check_cancel()
    if not bool(mx.all(mx.isfinite(pixels)).item()):
        raise RuntimeError('Non-finite VAE output')
    array = np.rint(np.asarray(pixels[0])*255).astype(np.uint8)
    if array.shape != (height,width,3) or array.std()<1:
        raise RuntimeError(f'Invalid decoded image: {array.shape}, std={array.std()}')
    mark('vae_decode',begin)
    result = dict(backend='mlx-metal',mlx=mx.__version__,mflux=importlib.metadata.version('mflux'),
                  prompt=prompt,seed=seed,width=width,height=height,steps=steps,sigmas=schedule,
                  models=paths,tokenizer=str(Path(args.tokenizer).resolve()),
                  workflow=str(Path(args.workflow).resolve()),timings_seconds=timings,
                  total_seconds=round(time.perf_counter()-start,3),
                  mlx_peak_memory_bytes=mx.get_peak_memory(),output=str(output),
                  torch_imported='torch' in sys.modules,
                  memory_policy='all models resident until engine close',prompt_cache_hit=cache_hit,engine_load_seconds=engine.load_seconds,
                  wired_limit_bytes=engine.wired_bytes,
                  weight_cache=engine.weight_cache,
                  weight_cache_root=engine.weight_cache_root,weight_cache_adapter=ADAPTER,
                  denoise_optimizations=[v for v in engine.denoise_optimizations if v!=engine.gemm_backend or tuned_for_shape],
                  gemm_backend=engine.gemm_backend if tuned_for_shape else 'mlx',
                  gemm_layers=engine.gemm_layers if tuned_for_shape else 0,
                  step_cache=engine._active_step_cache.stats() if engine._active_step_cache is not None else {'enabled':False},
                  rng='mlx.random; high 32 seed bits folded in when nonzero',quantization='INT8 codes preserved; BF16 affine scales and activations')
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('mqwen',json.dumps(result,ensure_ascii=False))
    Image.fromarray(array).save(output,pnginfo=metadata)
    output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'[mqwen] saved {output}',flush=True)
    return result


class GenerationCancelled(Exception):
    pass


class NativeEngine:
    """Owned by one inference thread. Model weights remain live between requests."""
    def __init__(self,args,log=print):
        self.log = log
        self.denoise_optimizations=['shared_convrot_constants']
        started = time.perf_counter()
        if not mx.metal.is_available():
            raise ValueError('MLX Metal device is unavailable')
        plan = Workflow(args.workflow).plan()
        validate_plan(plan)
        self.model_names = plan["models"]
        self.paths = model_paths(plan,[args.models]+args.extra_models,
                                 {r:getattr(args,r) for r in ('dit','text_encoder','vae')})
        self.tokenizer = QwenTokenizer(args.tokenizer)
        mx.set_cache_limit(512*1024**2)
        cache_root=getattr(args,'weight_cache',DEFAULT_CACHE)
        self.weight_cache_root=str(Path(cache_root).expanduser().resolve()) if cache_root is not None else None
        self.weight_cache={}
        self.text,self.weight_cache['text_encoder'] = load_or_convert(
            lambda:LanguageModel(TEXT_CONFIG,text_mode=False),self.paths['text_encoder'],'text_encoder',text_key,root=cache_root,log=log)
        self.transformer,self.weight_cache['dit'] = load_or_convert(
            Qwen21Transformer,self.paths['dit'],'dit',dit_key,root=cache_root,log=log)
        self.transformer.use_packed_projections = False
        self.vae,self.weight_cache['vae'] = load_or_convert(
            Qwen21VAE,self.paths['vae'],'vae',vae_key,root=cache_root,vae=True,log=log)
        mx.eval(self.text.parameters(),self.transformer.parameters(),self.vae.parameters())
        groups={layer.group for model in (self.text,self.transformer) for _,layer in model.named_modules()
                if isinstance(layer,(ConvRotLinear,ConvRotEmbedding)) and layer.group}
        prime_rotations(groups)
        from .metal_gemm import configure
        self.gemm_backend,self.gemm_layers=configure(self.transformer,mx.device_info(),
            mode=getattr(args,'gemm_backend','auto'),log=log)
        if self.gemm_layers:self.denoise_optimizations.append(self.gemm_backend)
        self.prompts = {}
        self._step_cache = self._active_step_cache = None
        self.wired_bytes = 0
        self._wired_previous = None
        if getattr(args,'wired_memory','auto') == 'auto':
            budget = residency_budget(mx.device_info())
            if budget:
                try:
                    self._wired_previous = mx.set_wired_limit(budget)
                    self.wired_bytes = budget
                    log(f'[mqwen] Metal residency budget: {budget/1024**3:.2f} GiB',flush=True)
                except RuntimeError as exc:
                    log(f'[mqwen] Metal residency unavailable; continuing: {exc}',flush=True)
        self.load_seconds = round(time.perf_counter()-started,3)
        log(f'[mqwen] resident engine ready: {self.load_seconds}s; {mx.get_active_memory()/1024**3:.2f} GiB',flush=True)

    def generate(self,args,cancelled=lambda:False):
        if not getattr(args,'step_cache',False):
            return _generate(args,self,cancelled)
        if getattr(args,'steps',8) != 8:
            raise ValueError('Experimental step cache requires the standard 8-step schedule')
        from .step_cache import CachedImageStep
        if self._step_cache is None:
            self._step_cache = CachedImageStep(self.transformer)
        cache = self._step_cache
        cache.reset()
        original = self.transformer._image_step_fn
        self._active_step_cache = cache
        self.transformer._image_step_fn = cache
        try:
            return _generate(args,self,cancelled)
        finally:
            self.transformer._image_step_fn = original
            self._active_step_cache = None
            cache.reset()

    def close(self):
        self.prompts.clear()
        self.transformer.clear_text_cache()
        self.transformer._step_fn = self.transformer._image_step_fn = None
        self.transformer._geometry_cache.clear()
        self._step_cache = self._active_step_cache = None
        self.text = self.transformer = self.vae = self.tokenizer = None
        clear_rotations()
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        if self._wired_previous is not None:
            mx.set_wired_limit(self._wired_previous)
            self._wired_previous = None
        self.log(f'[mqwen] engine released; active memory {mx.get_active_memory()/1024**3:.3f} GiB',flush=True)


def generate_native(args):
    engine = NativeEngine(args)
    try:
        return engine.generate(args)
    finally:
        engine.close()
