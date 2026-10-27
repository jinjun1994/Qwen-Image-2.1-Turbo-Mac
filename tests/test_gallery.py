import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
from mqwen.gallery import trash_image
from mqwen.web import Jobs

class GalleryTests(unittest.TestCase):
    def make_pair(self,root):
        image=root/'sample.png';image.write_bytes(b'original PNG bytes')
        image.with_suffix('.json').write_text('{"seed":42}')
        return image

    def test_trash_keeps_png_and_json_together(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as bin_folder:
            root=Path(folder).resolve();image=self.make_pair(root);target=Path(bin_folder)/'trashed'
            def move(path):shutil.move(path,target)
            with patch('send2trash.send2trash',side_effect=move):self.assertTrue(trash_image(image,root))
            self.assertEqual((target/'sample.png').read_bytes(),b'original PNG bytes')
            self.assertEqual(json.loads((target/'sample.json').read_text()),{'seed':42})
            self.assertEqual(list(root.iterdir()),[])

    def test_trash_failure_restores_both_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();image=self.make_pair(root)
            with patch('send2trash.send2trash',side_effect=OSError('Trash unavailable')):
                with self.assertRaises(OSError):trash_image(image,root)
            self.assertEqual(image.read_bytes(),b'original PNG bytes')
            self.assertEqual(json.loads(image.with_suffix('.json').read_text()),{'seed':42})
            self.assertEqual(len(list(root.iterdir())),2)

    def test_missing_sidecar_and_missing_image(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as bin_folder:
            root=Path(folder).resolve();image=self.make_pair(root);image.with_suffix('.json').unlink()
            with patch('send2trash.send2trash',side_effect=lambda path:shutil.move(path,Path(bin_folder)/'trashed')):
                self.assertTrue(trash_image(image,root))
                self.assertFalse(trash_image(image,root))

    def test_reject_symlinks_and_paths_outside_output(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as other:
            root=Path(folder).resolve();outside=self.make_pair(Path(other).resolve())
            link=root/'sample.png';link.symlink_to(outside)
            for image in [outside,link]:
                with self.subTest(image=image),self.assertRaises(ValueError):trash_image(image,root)
            self.assertTrue(outside.exists())

    def test_running_job_is_protected_and_success_removes_history(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as bin_folder:
            root=Path(folder).resolve();image=self.make_pair(root)
            jobs=Jobs.__new__(Jobs);jobs.lock=threading.Lock();jobs.output=root;jobs.active='one';jobs.jobs={'one':dict(status='running',image=str(image))}
            with self.assertRaises(RuntimeError):jobs.delete('one')
            self.assertTrue(image.exists())
            jobs.active=None;jobs.jobs['one']['status']='completed'
            with patch('send2trash.send2trash',side_effect=lambda path:shutil.move(path,Path(bin_folder)/'trashed')):self.assertTrue(jobs.delete('one')['trashed'])
            self.assertEqual(jobs.jobs,{})
            with self.assertRaises(FileNotFoundError):jobs.delete('one')
