import json
import struct
import tempfile
import unittest
from pathlib import Path
from mqwen.weights import WeightFile
from mqwen.workflow import Workflow
from mqwen.cli import audit

ROOT = Path(__file__).resolve().parents[1]

class WeightTests(unittest.TestCase):
    def make_file(self, directory, offsets=(0, 4), payload=b'abcd'):
        path = Path(directory) / 'test.safetensors'
        header = json.dumps({'x': {'dtype': 'U8', 'shape': [4], 'data_offsets': offsets}}).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header + payload)
        return path

    def test_readonly_mmap(self):
        with tempfile.TemporaryDirectory() as d:
            weights = WeightFile(self.make_file(d))
            with weights:
                view = weights.view('x')
                try:
                    self.assertEqual(bytes(view), b'abcd')
                    self.assertTrue(view.readonly)
                    with self.assertRaises(TypeError):
                        view[0] = 0
                finally:
                    view.release()
            self.assertIsNone(weights._mapping)

    def test_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'outside'):
                WeightFile(self.make_file(d, (1, 5)))

    def test_unindexed_data(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'trailing'):
                WeightFile(self.make_file(d, payload=b'abcde'))

    def test_truncated_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad'
            path.write_bytes(b'x')
            with self.assertRaises(ValueError):
                WeightFile(path)

class WorkflowTests(unittest.TestCase):
    def workflow(self):
        return Workflow(ROOT / 'workflows/turbo_t2i.json')

    def test_actual_workflow(self):
        plan = self.workflow().plan()
        self.assertEqual(plan['steps'], 8)
        self.assertEqual(plan['cfg'], 1)
        self.assertEqual(plan['sigmas'][-1], 0)
        self.assertEqual(plan['resolution_selector']['megapixels'], 0.2)
        self.assertEqual(plan['text_encoding_resolution'], 1024)
        self.assertIsInstance(plan['prompt'], str)

    def test_disabled_prompt_enhancer_not_executed(self):
        workflow = self.workflow()
        workflow.nodes['459:471']['inputs'] = {'untrusted': 'ignored'}
        self.assertIsInstance(workflow.plan()['prompt'], str)

    def test_enabled_enhancer_fails_explicitly(self):
        workflow = self.workflow()
        workflow.nodes['459:474']['inputs']['switch'] = True
        with self.assertRaisesRegex(ValueError, 'unsupported executor'):
            workflow.plan()

    def test_cycle_rejected(self):
        workflow = self.workflow()
        workflow.nodes['459:472']['inputs']['source'] = ['459:472', 0]
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            workflow.plan()

    def test_nan_schedule_rejected(self):
        workflow = self.workflow()
        workflow.nodes['459:484']['inputs']['sigmas'] = '1,nan,0'
        with self.assertRaises(ValueError):
            workflow.plan()

    def test_missing_weights_not_substituted(self):
        with tempfile.TemporaryDirectory() as d:
            result = audit(ROOT / 'workflows/turbo_t2i.json', d, {})
            self.assertEqual(len(result['issues']), 3)
            self.assertFalse(result['inference_ready'])

if __name__ == '__main__':
    unittest.main()
