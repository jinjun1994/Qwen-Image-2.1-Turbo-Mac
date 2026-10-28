import os
os.environ["USE_TORCH"]="0"
os.environ["USE_TF"]="0"
import importlib.util
from io import BytesIO
import tempfile
from pathlib import Path
import unittest

try:
    from PIL import Image
except ImportError:
    Image=None
from mqwen.inputs import store_upload,input_path
from mqwen.web import request_parameters

@unittest.skipUnless(Image is not None,'Pillow optional native dependency')
class UploadTests(unittest.TestCase):
    def test_normalizes_and_resolves_only_opaque_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            payload=BytesIO();Image.new('RGBA',(32,32),(20,50,80,99)).save(payload,format='PNG')
            result=store_upload(folder,payload.getvalue())
            self.assertEqual((result['width'],result['height']),(32,32))
            with Image.open(input_path(folder,result['id'])) as image:self.assertEqual(image.getpixel((0,0)),(20,50,80,99))
            for key in ('../secret','/tmp/file','a'*32):
                with self.assertRaises(ValueError):input_path(folder,key)
            target=input_path(folder,result['id']);alias=Path(folder)/'.inputs'/('b'*32+'.png');alias.symlink_to(target)
            with self.assertRaises(ValueError):input_path(folder,'b'*32)

    def test_rejects_invalid_or_unbounded_images(self):
        with tempfile.TemporaryDirectory() as folder:
            for payload in (b'',b'not a PNG',b'x'*(20*1024**2+1)):
                with self.assertRaises(ValueError):store_upload(folder,payload)
            data=BytesIO();Image.new('RGB',(1000,10)).save(data,format='PNG')
            with self.assertRaises(ValueError):store_upload(folder,data.getvalue())

    def test_edit_request_guards_and_legacy_defaults(self):
        data=dict(prompt='Change the cup',seed='42',mode='resolution',width=512,height=512)
        self.assertEqual(request_parameters(data)['output_format'],'rgb')
        self.assertEqual(request_parameters({**data,'references':['a'*32],'output_format':'rgba'})['references'],['a'*32])
        for changes in ({'mask':'a'*32},{'references':['../x']},{'references':['a'*32]*11},{'output_format':'jpg'},{'references':['a'*32],'step_cache':True}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):request_parameters({**data,**changes})

@unittest.skipUnless(Image is not None and importlib.util.find_spec('mlx'),'MLX and Pillow optional')
class EditingMathTests(unittest.TestCase):
    def test_composite_preserves_all_unmasked_rgba_bytes(self):
        import numpy as np
        from mqwen.editing import composite_mask
        original=np.random.default_rng(42).integers(0,256,(32,32,4),dtype=np.uint8)
        mask=Image.new('L',(32,32),0);mask.paste(255,(8,8,24,24))
        edited=Image.new('RGBA',(32,32),(255,0,0,150))
        result=np.asarray(composite_mask(edited,Image.fromarray(original),mask))
        np.testing.assert_array_equal(result[np.asarray(mask)==0],original[np.asarray(mask)==0])
        self.assertEqual(tuple(result[12,12]),(255,0,0,150))

    def test_vision_patch_order_matches_upstream_processor(self):
        import numpy as np
        from mqwen.editing import vision_patches,ASSETS
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        import json
        config=json.loads((ASSETS/'qwen21_processor.json').read_text())['image_processor']
        processor=Qwen2VLImageProcessorPil(**{k:v for k,v in config.items() if k!='image_processor_type'})
        image=Image.fromarray(np.random.default_rng(5).integers(0,256,(256,320,4),dtype=np.uint8))
        white=Image.new('RGB',image.size,'white');white.paste(image,mask=image.getchannel('A'))
        actual,grid=vision_patches(image)
        expected=processor(images=[white],return_tensors='np')
        np.testing.assert_allclose(actual,expected['pixel_values'],atol=3e-7)
        np.testing.assert_array_equal(grid,expected['image_grid_thw'][0])
