"""Persistent desktop preferences and explicit, verified model downloads."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from .workflow import Workflow
from .weights import WeightFile
from . import __version__

ROLES = {'dit':'diffusion_models','text_encoder':'text_encoders','vae':'vae'}
DEFAULT_HOME = Path.home()/'Library/Application Support/MQwen'

class Preferences:
    def __init__(self, root=None):
        self.root=Path(root or DEFAULT_HOME).expanduser()
        self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'settings.json'
        self.names=Workflow(Path(__file__).parent/'assets/turbo_t2i.json').plan()['models']
        defaults={role:'' for role in ROLES}
        for role,folder in ROLES.items():
            for base in (Path.home()/'MQwen/models',Path.home()/'ComfyUI-Shared/models',Path.home()/'Documents/ComfyUI2/models'):
                candidate=base/folder/self.names[role]
                if candidate.is_file():defaults[role]=str(candidate);break
        defaults['download_dir']=str(Path.home()/'MQwen/models')
        try:
            saved=json.loads(self.path.read_text())
            defaults.update({k:v for k,v in saved.items() if k in defaults and isinstance(v,str)})
        except FileNotFoundError:pass
        self.values=defaults
        self.lock=threading.RLock()

    def snapshot(self):
        with self.lock:
            return dict(values=dict(self.values),models={r:dict(path=self.values[r],exists=bool(self.values[r]) and Path(self.values[r]).is_file(),filename=self.names[r]) for r in ROLES})

    def save(self,data):
        if set(data)-set(self.values):raise ValueError('Unknown setting')
        values=dict(self.values)
        for key,value in data.items():
            if not isinstance(value,str) or '\x00' in value or len(value)>4096:raise ValueError('Invalid path')
            value=value.strip()
            if value:
                path=Path(value).expanduser()
                if not path.is_absolute():raise ValueError('Use an absolute path')
                value=str(path)
            if key in ROLES and value:
                if not Path(value).is_file():raise ValueError(f'Model not found: {value}')
                WeightFile(value)  # Header validation; engine validates model compatibility.
            values[key]=value
        if not values['download_dir']:raise ValueError('Choose a download folder')
        with self.lock:
            temp=self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
            temp.replace(self.path)
            self.values=values
        return self.snapshot()

    def apply(self,args):
        with self.lock:
            for role in ROLES:setattr(args,role,self.values[role] or None)

class Downloads:
    def __init__(self,preferences):
        self.preferences=preferences
        self.lock=threading.Lock()
        self.task=None
        self.cancelled=threading.Event()
        self.thread=None
        self.catalog=json.loads((Path(__file__).parent/'assets/models.json').read_text())

    def snapshot(self):
        with self.lock:return dict(catalog=self.catalog,task=dict(self.task) if self.task else None)

    def start(self,data):
        role=data.get('role');source=data.get('source','huggingface')
        if role not in ROLES:raise ValueError('Unknown model role')
        item=self.catalog['models'][role]
        if source=='custom':
            url=data.get('url','');digest=data.get('sha256','').lower()
        elif source in item['urls']:
            url=item['urls'][source];digest=item['sha256']
        else:raise ValueError('Unknown download source')
        parsed=urlparse(url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:raise ValueError('Download URL must be a public HTTPS URL')
        if not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('Provide the file SHA-256 to verify the download')
        base=Path(self.preferences.values['download_dir']).expanduser()
        target=base/ROLES[role]/self.preferences.names[role]
        if target.exists():raise ValueError('Target model already exists; select it or change the download folder')
        with self.lock:
            if self.thread and self.thread.is_alive():raise RuntimeError('A download is already running')
            self.cancelled.clear()
            self.task=dict(role=role,source=source,status='downloading',received=0,total=item['bytes'] if source!='custom' else None,path=str(target),error=None)
            self.thread=threading.Thread(target=self._run,args=(url,digest,target),daemon=True)
            self.thread.start()
        return self.snapshot()

    def cancel(self):self.cancelled.set()

    def _update(self,**values):
        with self.lock:self.task.update(values)

    def _run(self,url,digest,target):
        partial=target.with_suffix(target.suffix+'.part')
        identity=target.with_suffix(target.suffix+'.download.json')
        try:
            target.parent.mkdir(parents=True,exist_ok=True)
            fingerprint=dict(url=url,sha256=digest)
            previous=json.loads(identity.read_text()) if identity.exists() else None
            if previous!=fingerprint:
                partial.unlink(missing_ok=True)
                identity.write_text(json.dumps(fingerprint))
            offset=partial.stat().st_size if partial.exists() else 0
            request=Request(url,headers={'User-Agent':f'MQwen/{__version__}','Range':f'bytes={offset}-'} if offset else {'User-Agent':f'MQwen/{__version__}'})
            with urlopen(request,timeout=30) as response:
                if urlparse(response.url).scheme!='https':raise ValueError('Non-HTTPS redirect rejected')
                if offset and response.status==206:
                    content_range=response.headers.get('Content-Range','')
                    if not content_range.startswith(f'bytes {offset}-'):raise ValueError('Invalid resume response')
                elif response.status==200:offset=0
                else:raise ValueError(f'Unexpected HTTP status {response.status}')
                length=response.headers.get('Content-Length')
                total=offset+int(length) if length else None
                self._update(total=total,received=offset)
                with partial.open('ab' if offset else 'wb') as file:
                    received=offset
                    while True:
                        if self.cancelled.is_set():self._update(status='cancelled');return
                        chunk=response.read(4*1024*1024)
                        if not chunk:break
                        file.write(chunk);received+=len(chunk)
                        self._update(received=received)
                if total and received!=total:raise ValueError('Download incomplete; retry to resume')
            self._update(status='verifying')
            hasher=hashlib.sha256()
            with partial.open('rb') as file:
                while chunk:=file.read(8*1024*1024):
                    if self.cancelled.is_set():self._update(status='cancelled');return
                    hasher.update(chunk)
            if hasher.hexdigest()!=digest:
                partial.unlink(missing_ok=True)
                raise ValueError('SHA-256 mismatch; corrupted download removed, please retry')
            WeightFile(partial)
            # Never overwrite an existing model, including one created during download.
            os.link(partial,target);partial.unlink();identity.unlink(missing_ok=True)
            self._update(status='completed')
        except Exception as exc:self._update(status='failed',error=str(exc))

    def close(self):
        self.cancel()
        if self.thread:self.thread.join(timeout=35)
