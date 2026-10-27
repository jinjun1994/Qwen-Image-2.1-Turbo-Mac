import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('gemm_walk',Path(__file__).resolve().parents[1]/'scripts/gemm_walk.py')
walk=importlib.util.module_from_spec(spec);spec.loader.exec_module(walk)


class WalkCoverageTests(unittest.TestCase):
    def test_every_output_tile_is_visited_once_including_rectangles_and_tails(self):
        for rows in (1,4,5,13,16,17,31,64):
            for columns in (1,3,64,192,193):
                expected={(r,c) for r in range(rows) for c in range(columns)}
                for name in walk.SCHEDULES:
                    count=walk.launch_spec(rows,columns,name)[-1]
                    actual=[walk.coordinate(i,rows,columns,name) for i in range(count)]
                    valid=[p for p in actual if p is not None]
                    self.assertEqual(len(valid),len(expected),(rows,columns,name))
                    self.assertEqual(set(valid),expected,(rows,columns,name))

    def test_invalid_configuration_is_rejected(self):
        for args in ((0,3,'row'),(3,0,'row'),(3,3,'unknown')):
            with self.assertRaises(ValueError):walk.launch_spec(*args)
        with self.assertRaises(ValueError):walk.coordinate(9,3,3,'row')


@unittest.skipUnless(importlib.util.find_spec('mlx'),'MLX optional dependency not installed')
class WalkNumericsTests(unittest.TestCase):
    def test_every_schedule_matches_existing_kernel_with_partial_rows(self):
        import mlx.core as mx
        from mqwen.native_weights import ConvRotLinear
        from mqwen.metal_gemm import matmul
        layer=ConvRotLinear(mx.random.randint(-128,128,(192,256),key=mx.random.key(17)).astype(mx.int8),
            mx.linspace(.001,.03,192)[:,None],{'format':'int8_tensorwise'})
        x=mx.random.normal((257,256),key=mx.random.key(42)).astype(mx.bfloat16)
        expected=matmul(x,layer);mx.eval(expected)
        for name in walk.SCHEDULES:
            actual=walk.matmul(x,layer,name)
            self.assertTrue(bool(mx.all(actual==expected).item()),name)
