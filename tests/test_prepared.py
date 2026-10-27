import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('mlx'),'MLX optional dependency not installed')
class PreparedWeightsTests(unittest.TestCase):
    def fixture(self,folder):
        import mlx.core as mx
        import mlx.nn as nn
        from mqwen.native_weights import load_module
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj=nn.Linear(128,4,bias=False)
                self.embed_tokens=nn.Embedding(4,128)
            def __call__(self,x):return self.proj(x)
        source=Path(folder)/'source.safetensors'
        config=json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=64)).encode()
        arrays={}
        for name in ('proj','embed_tokens'):
            arrays[name+'.weight']=mx.arange(512).reshape(4,128).astype(mx.int8)
            arrays[name+'.weight_scale']=mx.full((4,1),.125)
            arrays[name+'.comfy_quant']=mx.array(list(config),dtype=mx.uint8)
        mx.save_safetensors(str(source),arrays)
        return Tiny,source,load_module(Tiny(),source)

    def test_round_trip_preserves_parameters_and_execution_without_conversion(self):
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        from mqwen.prepared import save_prepared,load_prepared
        with tempfile.TemporaryDirectory() as folder:
            factory,source,original=self.fixture(folder)
            root=Path(folder)/'cache'
            info=save_prepared(original,source,'tiny',root)
            with patch('mqwen.native_weights.ConvRotLinear.__init__',side_effect=AssertionError('must not repack')):
                loaded=load_prepared(factory(),source,'tiny',root)
            a,b=dict(tree_flatten(original.parameters())),dict(tree_flatten(loaded.parameters()))
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                np.testing.assert_array_equal(np.asarray(a[key].astype(mx.float32)),np.asarray(b[key].astype(mx.float32)))
            x=mx.random.normal((2,128)).astype(mx.bfloat16)
            np.testing.assert_array_equal(np.asarray(original(x).astype(mx.float32)),np.asarray(loaded(x).astype(mx.float32)))
            ids=mx.array([[1,3]])
            np.testing.assert_array_equal(np.asarray(original.embed_tokens(ids).astype(mx.float32)),np.asarray(loaded.embed_tokens(ids).astype(mx.float32)))
            self.assertFalse(info['reused'])
            self.assertTrue(save_prepared(original,source,'tiny',root)['reused'])

    def test_source_change_is_a_miss(self):
        import os
        from mqwen.prepared import save_prepared,load_prepared
        with tempfile.TemporaryDirectory() as folder:
            factory,source,model=self.fixture(folder)
            root=Path(folder)/'cache';save_prepared(model,source,'tiny',root)
            stat=source.stat();os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
            self.assertIsNone(load_prepared(factory(),source,'tiny',root))

    def test_damaged_payload_falls_back_to_fresh_source_module(self):
        import mlx.core as mx
        from mqwen.prepared import save_prepared,load_or_convert
        with tempfile.TemporaryDirectory() as folder:
            factory,source,model=self.fixture(folder)
            root=Path(folder)/'cache';info=save_prepared(model,source,'tiny',root)
            payload=Path(info['path'])/'weights.safetensors'
            with payload.open('r+b') as f:f.truncate(payload.stat().st_size-1)
            loaded,status=load_or_convert(factory,source,'tiny',lambda k:k,root=root,log=lambda *a,**kw:None)
            self.assertEqual(status,'rejected')
            mx.eval(loaded.parameters())

    def test_failed_save_never_publishes_partial_cache(self):
        from mqwen.prepared import save_prepared
        with tempfile.TemporaryDirectory() as folder:
            _,source,model=self.fixture(folder)
            root=Path(folder)/'cache'
            with patch('mqwen.prepared.mx.save_safetensors',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):save_prepared(model,source,'tiny',root)
            self.assertEqual(list(root.iterdir()),[])

    def test_full_verification_detects_payload_change_with_preserved_timestamp(self):
        import os
        from mqwen.prepared import save_prepared,identity,_validate
        with tempfile.TemporaryDirectory() as folder:
            _,source,model=self.fixture(folder)
            root=Path(folder)/'cache';info=save_prepared(model,source,'tiny',root)
            cache=Path(info['path']);payload=cache/'weights.safetensors';stat=payload.stat()
            with payload.open('r+b') as f:
                f.seek(-1,2);value=f.read(1)[0];f.seek(-1,2);f.write(bytes([value^1]))
            os.utime(payload,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'SHA256'):
                _validate(cache,identity(source,'tiny'),verify=True)
