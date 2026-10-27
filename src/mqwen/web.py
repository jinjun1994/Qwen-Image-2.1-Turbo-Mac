"""Loopback WebUI with a single resident MLX inference worker."""
from __future__ import annotations
from collections import deque
import json
import math
import os
import signal
from pathlib import Path
import re
import secrets
import queue
import traceback
from types import SimpleNamespace
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from .generate import output_size


def request_parameters(data):
    prompt = data.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt)>12000:
        raise ValueError('请输入 1–12000 字符的提示词')
    raw_seed = data.get('seed', '')
    if raw_seed == '':
        seed = secrets.randbits(32)
    elif isinstance(raw_seed, str) and re.fullmatch(r'\d{1,20}', raw_seed):
        seed = int(raw_seed)
    else:
        raise ValueError('种子须为无符号整数，留空为随机')
    if not 0 <= seed < 2**64:
        raise ValueError('种子超出 64 位整数范围')
    mode = data.get('mode')
    if mode == 'pixels':
        mp = data.get('megapixels')
        ratio = data.get('ratio')
        if type(mp) not in (int, float) or not math.isfinite(mp) or not .1 <= mp <= 4:
            raise ValueError('总像素须为 0.1–4 MP')
        if ratio not in ('1:1','4:3','3:4','16:9','9:16','3:2','2:3'):
            raise ValueError('不支持的宽高比')
        width, height = output_size({'aspect_ratio':ratio,'megapixels':mp,'multiple':32})
    elif mode == 'resolution':
        width, height = output_size({}, data.get('width'), data.get('height'))
        if width is None or height is None:
            raise ValueError('请指定宽高')
    else:
        raise ValueError('请选择总像素或自定义分辨率')
    steps=data.get('steps',8)
    if type(steps) is not int or steps not in (4,6,8):
        raise ValueError('步数须为 4、6 或 8')
    step_cache=data.get('step_cache',False)
    if type(step_cache) is not bool or (step_cache and steps != 8):
        raise ValueError('跨步缓存须为布尔值，且仅支持 8 步模式')
    return dict(prompt=prompt.strip(), seed=seed, width=width, height=height,steps=steps,step_cache=step_cache)


class Jobs:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output_dir).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs = {}
        self.active = None
        self.pending = queue.Queue()
        self.cancel_event = threading.Event()
        self.engine_state = "loading" if getattr(args,"auto_start",True) else "stopped"
        self.stopping = False
        for path in sorted(self.output.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:50]:
            try:
                result = json.loads(path.read_text())
                png = path.with_suffix('.png')
                if result.get('backend') != 'mlx-metal' or not png.is_file() or png.is_symlink():
                    continue
                key = 'saved-'+path.stem
                self.jobs[key] = dict(id=key,status='completed',progress=100,stage='已完成',
                                     params={**{k:result[k] for k in ('prompt','seed','width','height')},'steps':result.get('steps',8),
                                             'step_cache':result.get('step_cache',{}).get('enabled',False)},
                                     created=path.stat().st_mtime,metadata=result,image=str(png),logs=[])
            except (OSError,ValueError,KeyError):
                continue

        self.worker = threading.Thread(target=self._worker,daemon=True)
        if getattr(args,"auto_start",True):self.pending.put(("start",))
        self.worker.start()

    def _log(self,line,flush=False):
        with self.lock:
            if self.active is None:
                print(line,flush=True)
                return
            job = self.jobs[self.active]
            job['logs'].append(line[-2000:])
            if job['status'] != 'running':
                return
            step = re.search(r'denoise (\d+)/(\d+):',line)
            if step:
                job.update(stage=f'去噪 {step[1]}/{step[2]}',progress=20+int(64*int(step[1])/int(step[2])))
            elif '[mqwen] text_encode:' in line:
                job.update(stage='复用常驻 DiT',progress=15)
            elif '[mqwen] dit_load:' in line:
                job.update(stage='去噪 0/'+str(job['params'].get('steps',8)),progress=20)
            elif '[mqwen] denoise:' in line:
                job.update(stage='VAE 解码',progress=90)

    def _worker(self):
        engine = None
        try:
            while True:
                item = self.pending.get()
                if item is None:
                    break
                operation = item[0]
                if operation == 'start':
                    try:
                        from .native import NativeEngine, GenerationCancelled
                        engine = NativeEngine(self.args,log=self._log)
                        with self.lock:
                            if self.engine_state != 'stopping':
                                self.engine_state = 'ready'
                            self.engine_error = None
                    except Exception as exc:
                        self._log(traceback.format_exc())
                        with self.lock:
                            self.engine_state = 'failed'
                            self.engine_error = str(exc)
                    continue
                if operation == 'stop':
                    if engine is not None:
                        engine.close()
                        engine = None
                    with self.lock:
                        self.engine_state = 'stopped'
                    continue
                _,key,destination,params = item
                args = SimpleNamespace(**vars(self.args),**params,output=str(destination))
                try:
                    if engine is None:
                        raise RuntimeError('请先启动推理服务')
                    result = engine.generate(args,self.cancel_event.is_set)
                    with self.lock:
                        self.jobs[key].update(status='completed',stage='已完成',progress=100,metadata=result)
                except GenerationCancelled:
                    with self.lock:
                        self.jobs[key].update(status='cancelled',stage='已取消')
                except Exception as exc:
                    self._log(traceback.format_exc())
                    with self.lock:
                        self.jobs[key].update(status='failed',stage='生成失败',error=str(exc))
                finally:
                    with self.lock:
                        self.active = None
        finally:
            if engine is not None:
                engine.close()

    def start_engine(self):
        with self.lock:
            if self.stopping or self.engine_state == 'stopping':
                raise RuntimeError('服务正在关闭，请稍后再启动')
            if self.engine_state in ('ready','loading'):
                return
            self.engine_state = 'loading'
            self.engine_error = None
            self.pending.put(('start',))

    def stop_engine(self):
        with self.lock:
            if self.engine_state in ('stopped','stopping'):
                return
            self.engine_state = 'stopping'
            if self.active:
                self.jobs[self.active].update(status='cancelling',stage='正在取消并关闭推理服务')
            self.cancel_event.set()
            self.pending.put(('stop',))

    def shutdown(self):
        with self.lock:
            self.stopping = True
        self.cancel_event.set()
        self.pending.put(None)
        self.worker.join()

    def start(self, data):
        params = request_parameters(data)
        with self.lock:
            if self.stopping or self.engine_state != 'ready':
                raise RuntimeError(getattr(self,'engine_error',None) or '请等待推理服务就绪，或点击启动推理服务')
            if self.active is not None:
                raise RuntimeError('已有任务正在生成，请等待完成或取消')
            key = secrets.token_hex(12)
            destination = self.output/f'web-{key}.png'
            job = dict(id=key,status='running',progress=2,stage='准备模型',params=params,
                       created=time.time(),logs=deque(maxlen=160),image=str(destination))
            self.jobs[key] = job
            self.active = key
            self.cancel_event.clear()
            while len(self.jobs)>50:
                self.jobs.pop(min(self.jobs,key=lambda k:self.jobs[k]['created']))
            self.pending.put(('generate',key,destination,params))
            return key

    def cancel(self):
        with self.lock:
            if self.active is not None:
                self.jobs[self.active].update(status='cancelling',stage='正在取消，等待当前 GPU 运算结束')
                self.cancel_event.set()

    def state(self):
        with self.lock:
            jobs = []
            for job in sorted(self.jobs.values(),key=lambda j:j['created'],reverse=True):
                public = {k:v for k,v in job.items() if k not in ('image','logs','metadata')}
                public['logs'] = list(job['logs'])
                if job['status']=='completed':
                    public['image_url'] = '/image/'+job['id']
                    public['metadata_url'] = '/metadata/'+job['id']
                    public['timings'] = job['metadata'].get('timings_seconds',{})
                    public['total_seconds'] = job['metadata'].get('total_seconds')
                jobs.append(public)
            return dict(active=self.active,jobs=jobs,engine_state=self.engine_state,engine_error=getattr(self,'engine_error',None))


class Server(ThreadingHTTPServer):
    daemon_threads = True


def serve(args):
    from .product import Preferences, Downloads
    preferences=Preferences(getattr(args,'config_dir',None)) if getattr(args,'desktop',False) else None
    if preferences:preferences.apply(args)
    downloads=Downloads(preferences) if preferences else None
    jobs = None
    assets = Path(__file__).parent/'assets/web'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def reply(self,code,body,content_type='application/json; charset=utf-8'):
            if isinstance(body,dict):
                body = json.dumps(body,ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.allowed():
                return self.reply(403,{'error':'Localhost only'})
            route = urlparse(self.path).path
            if route=='/api/state':
                return self.reply(200,{**jobs.state(),'desktop':preferences is not None,'api_base':f'http://127.0.0.1:{self.server.server_port}'})
            if route=='/api/settings' and preferences:
                return self.reply(200,preferences.snapshot())
            if route=='/api/downloads' and downloads:
                return self.reply(200,downloads.snapshot())
            if route=='/api/health':
                return self.reply(200,{'service':'mqwen','pid':os.getpid(),'version':'0.3.0'})
            static = {'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if route in static:
                file,mime = static[route]
                return self.reply(200,(assets/file).read_bytes(),mime)
            if route.startswith(('/image/','/metadata/')):
                key = route.rsplit('/',1)[-1]
                with jobs.lock:
                    job = jobs.jobs.get(key)
                    if job is None or job['status']!='completed':
                        return self.reply(404,{'error':'Image not found'})
                    path = Path(job['image'])
                    metadata = job['metadata']
                if route.startswith('/metadata/'):
                    return self.reply(200,metadata)
                if path.is_file() and not path.is_symlink():
                    return self.reply(200,path.read_bytes(),'image/png')
            self.reply(404,{'error':'Not found'})

        def do_POST(self):
            origin = self.headers.get('Origin')
            expected = {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if not self.allowed() or (origin is not None and origin not in expected):
                return self.reply(403,{'error':'Local origin required'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536 or self.headers.get('Content-Type')!='application/json':
                    raise ValueError('Expected JSON request up to 64 KiB')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data,dict):
                    raise ValueError('Expected a JSON object')
                if self.path=='/api/settings' and preferences:
                    with jobs.lock:
                        if jobs.engine_state not in ('stopped','failed') or jobs.active:
                            raise RuntimeError('请先关闭推理服务再更改模型路径')
                        if downloads and downloads.thread and downloads.thread.is_alive():
                            raise RuntimeError('请先完成或取消下载')
                        saved=preferences.save(data)
                        preferences.apply(args)
                    return self.reply(200,saved)
                if self.path=='/api/downloads/start' and downloads:
                    with jobs.lock:
                        result=downloads.start(data)
                    return self.reply(202,result)
                if self.path=='/api/downloads/cancel' and downloads:
                    downloads.cancel()
                    return self.reply(202,{'ok':True})
                if self.path=='/api/server/shutdown':
                    token=os.environ.get('MQWEN_CONTROL_TOKEN')
                    if not token or self.headers.get('X-MQwen-Control')!=token:
                        return self.reply(403,{'error':'Desktop control token required'})
                    self.reply(202,{'ok':True})
                    threading.Thread(target=server.shutdown,daemon=True).start()
                    return
                if self.path=='/api/generate':
                    return self.reply(202,{'id':jobs.start(data)})
                if self.path=='/api/engine/start':
                    jobs.start_engine()
                    return self.reply(202,{'ok':True})
                if self.path=='/api/engine/stop':
                    jobs.stop_engine()
                    return self.reply(202,{'ok':True})
                if self.path=='/api/cancel':
                    jobs.cancel()
                    return self.reply(200,{'ok':True})
                self.reply(404,{'error':'Not found'})
            except (ValueError,TypeError,KeyError) as exc:
                self.reply(400,{'error':str(exc)})
            except RuntimeError as exc:
                self.reply(409,{'error':str(exc)})
            except OSError as exc:
                self.reply(500,{'error':str(exc)})
    server = Server(('127.0.0.1',args.port),Handler)
    jobs = Jobs(args)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM,lambda *_: threading.Thread(target=server.shutdown,daemon=True).start())
    print(f'MQwen WebUI: http://127.0.0.1:{server.server_port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        jobs.shutdown()
        if downloads:downloads.close()
        server.server_close()
