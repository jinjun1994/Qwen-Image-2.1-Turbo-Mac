import unittest
from mqwen.web import request_parameters

class WebParametersTests(unittest.TestCase):
    def payload(self,**updates):
        return dict(prompt='a red teapot',seed='42',mode='pixels',megapixels=1,ratio='1:1',**updates)

    def test_megapixel_size_aligned(self):
        p=request_parameters(self.payload())
        self.assertEqual((p['width'],p['height']),(992,992))
        self.assertEqual(p['seed'],42)

    def test_dimensions_and_64bit_seed(self):
        p=request_parameters(dict(prompt='test',seed=str(2**64-1),mode='resolution',width=1280,height=768))
        self.assertEqual(p['seed'],2**64-1)
        self.assertEqual(p['width'],1280)

    def test_reject_bad_inputs_before_start(self):
        invalid=[{'step_cache':'true'},{'step_cache':True,'steps':4},{'steps':5},{'steps':True},{'seed':str(2**64)},{'seed':1.5},{'seed':'-1'}, {'prompt':' '},
                 {'megapixels':float('nan')},{'megapixels':5},{'ratio':'100:1'},
                 {'mode':'resolution','width':1025,'height':1024},
                 {'mode':'resolution','width':True,'height':1024}]
        for updates in invalid:
            p=self.payload();p.update(updates)
            with self.subTest(updates=updates),self.assertRaises((ValueError,KeyError)):
                request_parameters(p)

    def test_random_seed(self):
        p=self.payload();p['seed']=''
        self.assertTrue(0<=request_parameters(p)['seed']<2**32)

class ResidentWorkerTests(unittest.TestCase):
    def test_worker_retains_engine_after_cancel_and_closes_on_shutdown(self):
        import tempfile,time,threading
        from types import SimpleNamespace
        from unittest.mock import patch
        from mqwen.web import Jobs
        from mqwen.native import GenerationCancelled
        entered=threading.Event()
        engines=[]
        class FakeEngine:
            def __init__(self,args,log):
                engines.append(self)
                self.calls=0
                self.closed=False
            def generate(self,args,cancelled):
                self.calls+=1
                entered.set()
                if self.calls==1:
                    for _ in range(200):
                        if cancelled():raise GenerationCancelled()
                        time.sleep(.005)
                    raise AssertionError('cancel was not delivered')
                return {'timings_seconds':{},'total_seconds':.1}
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as folder,patch('mqwen.native.NativeEngine',FakeEngine):
            jobs=Jobs(SimpleNamespace(output_dir=folder))
            data=dict(prompt='test',seed='42',mode='resolution',width=512,height=512)
            try:
                for _ in range(200):
                    if jobs.engine_state=='ready':break
                    time.sleep(.005)
                first=jobs.start(data)
                self.assertTrue(entered.wait(2))
                with self.assertRaises(RuntimeError):jobs.start(data)
                jobs.cancel()
                for _ in range(200):
                    if jobs.active is None:break
                    time.sleep(.005)
                self.assertEqual(jobs.jobs[first]['status'],'cancelled')
                second=jobs.start(data)
                for _ in range(200):
                    if jobs.active is None:break
                    time.sleep(.005)
                self.assertEqual(jobs.jobs[second]['status'],'completed')
                self.assertEqual(len(engines),1)
                self.assertEqual(engines[0].calls,2)
                self.assertFalse(engines[0].closed)
                jobs.stop_engine()
                for _ in range(200):
                    if jobs.engine_state=='stopped':break
                    time.sleep(.005)
                self.assertEqual(jobs.engine_state,'stopped')
                self.assertTrue(engines[0].closed)
                with self.assertRaises(RuntimeError):jobs.start(data)
                jobs.start_engine()
                for _ in range(200):
                    if jobs.engine_state=='ready':break
                    time.sleep(.005)
                self.assertEqual(jobs.engine_state,'ready')
                self.assertEqual(len(engines),2)

            finally:jobs.shutdown()
            self.assertTrue(all(engine.closed for engine in engines))
            self.assertFalse(jobs.worker.is_alive())
