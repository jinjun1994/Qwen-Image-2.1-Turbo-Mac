"""Bounded local image uploads. IDs never resolve arbitrary filesystem paths."""
from io import BytesIO
from pathlib import Path
import re
import secrets
import warnings

MAX_UPLOAD=20*1024**2
MAX_PIXELS=16*1024**2


def store_upload(output,payload):
    from PIL import Image,ImageOps,UnidentifiedImageError
    if not payload or len(payload)>MAX_UPLOAD:raise ValueError('Upload an image up to 20 MiB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in ('PNG','JPEG','WEBP'):raise ValueError('Use a PNG, JPEG or WebP image')
                if image.width*image.height>MAX_PIXELS or max(image.size)/min(image.size)>8:
                    raise ValueError('Images must be at most 16 MP with aspect ratio between 1:8 and 8:1')
                rgba=ImageOps.exif_transpose(image).convert('RGBA')
    except (OSError,UnidentifiedImageError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise ValueError('Could not decode the uploaded image') from exc
    root=Path(output).resolve()/'.inputs'
    if root.is_symlink():raise ValueError('Upload folder must not be a symlink')
    root.mkdir(exist_ok=True)
    key=secrets.token_hex(16)
    rgba.save(root/(key+'.png'))
    return dict(id=key,width=rgba.width,height=rgba.height,url='/input/'+key)


def input_path(output,key):
    if not isinstance(key,str) or not re.fullmatch('[a-f0-9]{32}',key):raise ValueError('Invalid uploaded image ID')
    root=Path(output).resolve()/'.inputs'
    if root.is_symlink():raise ValueError('Upload folder must not be a symlink')
    path=root/(key+'.png')
    if path.is_symlink() or not path.is_file():raise ValueError('Uploaded image not found; upload it again')
    return path
