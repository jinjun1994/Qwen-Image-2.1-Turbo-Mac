"""M3 Pro tensorwise INT8 GEMM using a specialized MLX Steel kernel (MIT).

The kernel preserves packed INT8 codes and BF16 affine decode; it changes scale
addressing and the MMA tile. MLX remains the fallback for unsupported inputs.
"""
from functools import lru_cache
from pathlib import Path
import mlx.core as mx

SHAPES = {(4096,4096),(4096,12288),(12288,4096)}  # (input, output)
BACKEND = 'metal-int8-tensorwise-64x64x32'


@lru_cache(maxsize=1)
def _kernel():
    header=(Path(__file__).parent/'assets/kernels/int8_tensorwise.metal').read_text()
    return mx.fast.metal_kernel(name='mqwen_tensorwise_qmm',input_names=['x','w','s','b'],output_names=['y'],header=header,source='''
    threadgroup T Xs[BM*(BK+16/sizeof(T))];
    threadgroup T Ws[BN*(BK+16/sizeof(T))];
    qmm_t_impl<T,64,8,true,T,BM,BK,BN>(w,s,b,x,y,Xs,Ws,K,N,M,K,
        threadgroup_position_in_grid,thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
    ''')


def supported(layer,x):
    return (x.dtype==mx.bfloat16 and x.ndim>=2 and x.shape[-1]==layer.in_features
            and 256<=x.size//layer.in_features<=4096
            and (layer.in_features,layer.out_features) in SHAPES)


def matmul(x,layer):
    """Caller verifies tensorwise row scales once, before compiling generation."""
    if x.ndim<2 or x.dtype!=mx.bfloat16 or x.shape[-1]<1 or x.shape[-1]!=layer.in_features:
        raise ValueError('Tuned GEMM requires rank >= 2 BF16 activations with matching input dimension')
    m=x.size//x.shape[-1];k=x.shape[-1];n=layer.out_features
    if m<1 or k%64 or n%64:
        raise ValueError('Tuned GEMM requires nonempty rows and K/N aligned to 64')
    return _kernel()(inputs=[x,layer.weight,layer.scales[:,:1].reshape(-1),layer.biases[:,:1].reshape(-1)],
        template=[('T',x.dtype),('M',m),('N',n),('K',k),('BM',64),('BN',64),('BK',32)],
        grid=((n+63)//64*128,(m+63)//64,1),threadgroup=(128,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[x.dtype])[0]


def configure(model,device,*,mode='auto',log=print):
    """Enable only on the validated device/version and verified row-scale layers."""
    from .native_weights import ConvRotLinear
    layers=[l for _,l in model.named_modules() if type(l) is ConvRotLinear]
    for l in layers:l._tuned_gemm=False
    if (mode!='auto' or device.get('device_name')!='Apple M3 Pro'
            or device.get('architecture')!='applegpu_g15s' or mx.__version__!='0.32.4'):
        return 'mlx',0
    candidates=[l for l in layers if (l.in_features,l.out_features) in SHAPES and l.group_size==64]
    if not candidates:return 'mlx',0
    try:
        checks=[mx.all(l.scales==l.scales[:,:1]) & mx.all(l.biases==l.biases[:,:1]) for l in candidates]
        mx.eval(checks)
        candidates=[l for l,check in zip(candidates,checks) if bool(check.item())]
        probes={}
        for l in candidates:probes.setdefault((l.in_features,l.out_features),l)
        # Compile and compare each geometry before enabling any layer. New shapes
        # still specialize M at first use; unsupported inputs take the MLX path.
        for l in probes.values():
            x=mx.random.normal((1,1024,l.in_features),key=mx.random.key(42)).astype(mx.bfloat16)
            expected=mx.quantized_matmul(x,l.weight,l.scales,l.biases,transpose=True,group_size=64,bits=8)
            actual=matmul(x,l)
            if not bool(mx.all(actual==expected).item()):
                raise ValueError('GEMM preflight did not match the MLX reference exactly')
    except (RuntimeError,ValueError,OSError) as exc:
        log(f'[mqwen] tuned GEMM unavailable; using MLX: {exc}',flush=True)
        return 'mlx',0
    for l in candidates:l._tuned_gemm=True
    if not candidates:return 'mlx',0
    log(f'[mqwen] tuned Metal GEMM enabled: {len(candidates)} DiT projections; tile=64x64x32',flush=True)
    return BACKEND,len(candidates)
