import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from mqwen.product import Preferences,Downloads


def weights():
    header=json.dumps({'a':{'dtype':'U8','shape':[4],'data_offsets':[0,4]}}).encode()
    return struct.pack('<Q',len(header))+header+b'abcd'

class Response(io.BytesIO):
    status=200
    url='https://example.org/model.safetensors'
    def __init__(self,data,*,offset=None):
        super().__init__(data)
        self.headers={'Content-Length':str(len(data))}
        if offset is not None:
            self.status=206;self.headers['Content-Range']=f'bytes {offset}-{offset+len(data)-1}/{offset+len(data)}'

class ProductTests(unittest.TestCase):
    def test_preferences_roundtrip_and_invalid_model(self):
        with tempfile.TemporaryDirectory() as root:
            prefs=Preferences(root);model=Path(root)/'model.safetensors';model.write_bytes(weights())
            prefs.save({'dit':str(model),'download_dir':root})
            self.assertEqual(Preferences(root).values['dit'],str(model))
            with self.assertRaises(ValueError):prefs.save({'dit':'relative/path'})
            model.write_bytes(b'not weights')
            with self.assertRaises(ValueError):prefs.save({'dit':str(model)})
            with self.assertRaises(ValueError):prefs.save({'unknown':'x'})

    def run_download(self,root,*,digest=None,response=None,offset=0):
        prefs=Preferences(Path(root)/'config');prefs.save({'download_dir':str(Path(root)/'models')})
        manager=Downloads(prefs);payload=weights();url='https://example.org/model.safetensors'
        digest=digest or hashlib.sha256(payload).hexdigest()
        target=Path(prefs.values['download_dir'])/'vae'/prefs.names['vae']
        if offset:
            target.parent.mkdir(parents=True)
            target.with_suffix(target.suffix+'.part').write_bytes(payload[:offset])
            target.with_suffix(target.suffix+'.download.json').write_text(json.dumps(dict(url=url,sha256=digest)))
        with patch('mqwen.product.urlopen',return_value=response or Response(payload)) as opened:
            manager.start(dict(role='vae',source='custom',url=url,sha256=digest));manager.thread.join(3)
            self.assertFalse(manager.thread.is_alive())
        return manager,target,opened

    def test_download_verified_without_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            manager,target,_=self.run_download(root)
            self.assertEqual(manager.snapshot()['task']['status'],'completed')
            self.assertEqual(target.read_bytes(),weights())
            with self.assertRaises(ValueError):manager.start(dict(role='vae',source='huggingface'))

    def test_resume_range_and_server_ignores_range(self):
        for supports_range in [True,False]:
            with self.subTest(supports_range=supports_range),tempfile.TemporaryDirectory() as root:
                offset=12
                response=Response(weights()[offset:],offset=offset) if supports_range else Response(weights())
                manager,target,opened=self.run_download(root,response=response,offset=offset)
                self.assertEqual(target.read_bytes(),weights())
                self.assertEqual(opened.call_args.args[0].get_header('Range'),'bytes=12-')
                self.assertEqual(manager.snapshot()['task']['status'],'completed')

    def test_bad_hash_and_reject_non_https(self):
        with tempfile.TemporaryDirectory() as root:
            manager,target,_=self.run_download(root,digest='0'*64)
            self.assertEqual(manager.snapshot()['task']['status'],'failed')
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(target.suffix+'.part').exists())
            with self.assertRaises(ValueError):manager.start(dict(role='vae',source='custom',url='http://example.org/x',sha256='0'*64))
