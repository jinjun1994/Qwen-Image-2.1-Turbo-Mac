import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('mlx'), 'MLX optional dependency not installed')
class NativeTests(unittest.TestCase):
    def test_compiled_shared_rotation_matches_original_dense_math(self):
        import mlx.core as mx
        import numpy as np
        from mqwen.native_weights import rotate,hadamard,prime_rotations,clear_rotations
        prime_rotations((64,256))
        try:
            for group in (64,256):
                compiled=mx.compile(lambda x:rotate(x,group))
                for dtype in (mx.float32,mx.bfloat16):
                    x=mx.random.normal((2,group,2),key=mx.random.key(42)).astype(dtype).transpose(0,2,1)
                    expected=(x.reshape(-1,group)@hadamard(group).astype(dtype)).reshape(x.shape)
                    np.testing.assert_array_equal(np.asarray(compiled(x).astype(mx.float32)),np.asarray(expected.astype(mx.float32)))
        finally:clear_rotations()

    def test_step_cache_guards(self):
        from mqwen.step_cache import can_reuse
        for step in (0,1,6,7):
            self.assertFalse(can_reuse(step,8,.01,True,False,.08))
        self.assertTrue(can_reuse(2,8,.07,True,False,.08))
        self.assertFalse(can_reuse(2,8,.09,True,False,.08))
        self.assertFalse(can_reuse(2,8,.01,True,True,.08))
        self.assertFalse(can_reuse(2,8,.01,False,False,.08))
        self.assertFalse(can_reuse(2,8,float('nan'),True,False,.08))

    def test_step_cache_restored_after_cancellation(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from mqwen.native import NativeEngine,GenerationCancelled
        original=object()
        class Cache:
            def reset(self):self.empty=True
        engine=NativeEngine.__new__(NativeEngine)
        engine.transformer=SimpleNamespace(_image_step_fn=original)
        engine._step_cache=Cache()
        engine._active_step_cache=None
        with patch('mqwen.native._generate',side_effect=GenerationCancelled):
            with self.assertRaises(GenerationCancelled):
                engine.generate(SimpleNamespace(step_cache=True,steps=8))
        self.assertIs(engine.transformer._image_step_fn,original)
        self.assertIsNone(engine._active_step_cache)
        self.assertTrue(engine._step_cache.empty)

    def test_rotation_is_orthogonal_and_self_inverse(self):
        import mlx.core as mx
        import numpy as np
        from mqwen.native_weights import hadamard,rotate
        h = np.asarray(hadamard(256))
        np.testing.assert_allclose(h@h.T,np.eye(256),atol=1e-6)
        x = mx.random.normal((2,512))
        np.testing.assert_allclose(np.asarray(rotate(rotate(x,256),256)),np.asarray(x),atol=2e-6)

    def test_int8_linear_preserves_signed_codes(self):
        import mlx.core as mx
        import numpy as np
        from mqwen.native_weights import ConvRotLinear,rotate
        q = mx.array(np.arange(-128,128,dtype=np.int8).reshape(4,64))
        scale = mx.full((4,1),0.125,dtype=mx.float32)
        layer = ConvRotLinear(q,scale,{'format':'int8_tensorwise','convrot':True,'convrot_groupsize':64})
        x = mx.random.normal((3,64)).astype(mx.bfloat16)
        out = layer(x).astype(mx.float32)
        expected = rotate(x,64).astype(mx.float32) @ (q.astype(mx.float32)*scale).T
        np.testing.assert_allclose(np.asarray(out),np.asarray(expected),rtol=.03,atol=.3)

    def test_embedding_undoes_rotation(self):
        import mlx.core as mx
        import numpy as np
        from mqwen.native_weights import ConvRotEmbedding,rotate
        q = mx.array(np.arange(-128,128,dtype=np.int8).reshape(4,64))
        scale = mx.full((4,1),.125)
        layer = ConvRotEmbedding(q,scale,{'convrot':True,'convrot_groupsize':64})
        indices = mx.array([[1,3]])
        expected = rotate(q[indices].astype(mx.float32)*scale[indices],64).astype(mx.bfloat16)
        np.testing.assert_array_equal(np.asarray(layer(indices).astype(mx.float32)),np.asarray(expected.astype(mx.float32)))

    def test_vae_mapping(self):
        from mqwen.native_weights import vae_key
        self.assertEqual(vae_key('decoder.upsamples.2.upsamples.0.residual.6.weight'),
                         'decoder.up_blocks.2.resnets.0.conv2.weight')
        self.assertEqual(vae_key('encoder.downsamples.0.downsamples.2.resample.1.weight'),
                         'encoder.down_blocks.0.downsampler.resample.1.weight')
        self.assertIsNone(vae_key('decoder.upsamples.0.upsamples.3.time_conv.weight'))

@unittest.skipUnless(importlib.util.find_spec('tokenizers'), 'tokenizers optional dependency not installed')
class TokenizerTests(unittest.TestCase):
    def test_known_qwen_token_ids(self):
        from mqwen.tokenizer import QwenTokenizer
        from pathlib import Path
        tokenizer = QwenTokenizer(Path(__file__).resolve().parents[1]/'src/mqwen/assets/tokenizer')
        self.assertEqual(tokenizer.encode('A red teapot.'),[32,2518,1013,89901,13])
        self.assertEqual(tokenizer.encode('一只红色茶壶。'),[102292,104165,100019,106469,1773])
        self.assertEqual(tokenizer.encode('<|im_start|>user\nHello!<|im_end|>\n'),
                         [151644,872,198,9707,0,151645,198])

class PlanTests(unittest.TestCase):
    def test_residency_budget_leaves_system_headroom(self):
        from mqwen.memory import residency_budget
        self.assertEqual(residency_budget({'memory_size':100,'max_recommended_working_set_size':80}),60)
        self.assertEqual(residency_budget({'memory_size':100,'max_recommended_working_set_size':40}),40)
        self.assertEqual(residency_budget({}),0)

    def test_workflow_output_policy(self):
        from mqwen.generate import output_size
        self.assertEqual(output_size({'aspect_ratio':'1:1 (Square)','megapixels':.2,'multiple':8}),(448,448))
        with self.assertRaises(ValueError):
            output_size({},width=1024)
        with self.assertRaises(ValueError):
            output_size({},width=1025,height=1024)
