import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('mlx'),'MLX optional dependency not installed')
class MetalGemmTests(unittest.TestCase):
    def layer(self):
        import mlx.core as mx
        from mqwen.native_weights import ConvRotLinear
        # Each row has a different scale, exercising specialized row addressing.
        codes=(mx.arange(64*256).reshape(64,256)*17+mx.arange(64)[:,None]*13).astype(mx.int8)
        return ConvRotLinear(codes,mx.linspace(.001,.03,64)[:,None],{'format':'int8_tensorwise'})

    def test_mma_tail_rows_and_noncontiguous_input_match_dense_decode(self):
        import mlx.core as mx
        import numpy as np
        from mqwen.metal_gemm import matmul
        l=self.layer()
        x=mx.random.normal((257,2,256),key=mx.random.key(42)).astype(mx.bfloat16)[:,1,:]
        decoded=mx.dequantize(l.weight,l.scales,l.biases,group_size=64,bits=8)
        expected=(x.astype(mx.float32)@decoded.astype(mx.float32).T).astype(mx.bfloat16)
        y=mx.compile(lambda x:matmul(x,l))(x)
        np.testing.assert_allclose(np.asarray(y.astype(mx.float32)),np.asarray(expected.astype(mx.float32)),rtol=.008,atol=1e-3)

    def test_unaligned_output_cannot_reach_aligned_shader(self):
        import mlx.core as mx
        from mqwen.metal_gemm import matmul
        with self.assertRaises(ValueError):
            matmul(mx.zeros((257,256),dtype=mx.bfloat16),SimpleNamespace(in_features=256,out_features=65))

    def test_dispatch_rejects_unvalidated_shapes_and_dtypes(self):
        import mlx.core as mx
        from mqwen.metal_gemm import supported
        l=SimpleNamespace(in_features=4096,out_features=4096)
        self.assertFalse(supported(l,mx.zeros((255,4096),dtype=mx.bfloat16)))
        self.assertFalse(supported(l,mx.zeros((4097,4096),dtype=mx.bfloat16)))
        self.assertFalse(supported(l,mx.zeros((1024,4096),dtype=mx.float32)))
        self.assertTrue(supported(l,mx.zeros((257,4096),dtype=mx.bfloat16)))

    def test_unsupported_device_and_explicit_mlx_do_not_compile(self):
        import mlx.nn as nn
        from mqwen.metal_gemm import configure
        model=nn.Sequential(self.layer())
        with patch('mqwen.metal_gemm.matmul',side_effect=AssertionError('must not compile')):
            self.assertEqual(configure(model,{'device_name':'Apple M2'}),('mlx',0))
            self.assertEqual(configure(model,{'device_name':'Apple M3 Pro','architecture':'applegpu_g15s'},mode='mlx'),('mlx',0))
        self.assertFalse(model.layers[0]._tuned_gemm)

    def test_failed_preflight_leaves_every_layer_on_mlx(self):
        import mlx.nn as nn
        from mqwen.metal_gemm import configure
        model=nn.Sequential(self.layer(),self.layer());logs=[]
        with patch('mqwen.metal_gemm.SHAPES',{(256,64)}),patch('mqwen.metal_gemm.matmul',side_effect=RuntimeError('test compile failure')):
            self.assertEqual(configure(model,{'device_name':'Apple M3 Pro','architecture':'applegpu_g15s'},log=lambda x,**kw:logs.append(x)),('mlx',0))
        self.assertTrue(any('test compile failure' in line for line in logs))
        self.assertTrue(all(not l._tuned_gemm for l in model.layers))

    def test_groupwise_scales_are_not_misread_as_tensorwise(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mqwen.metal_gemm import configure
        l=self.layer();l.scales=l.scales+mx.arange(4)[None,:].astype(mx.bfloat16)
        with patch('mqwen.metal_gemm.SHAPES',{(256,64)}),patch('mqwen.metal_gemm.matmul',side_effect=AssertionError('must reject groupwise scales')):
            self.assertEqual(configure(nn.Sequential(l),{'device_name':'Apple M3 Pro','architecture':'applegpu_g15s'}),('mlx',0))
        self.assertFalse(l._tuned_gemm)
